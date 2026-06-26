import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import Client, create_client

from os import environ


SUPABASE_URL = environ["SUPABASE_URL"]
SUPABASE_KEY = environ["SUPABASE_KEY"]
DATASET_BUCKET = "dataset"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Gerra Samples API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    user_id: str
    username: str
    first_name: str
    last_name: str
    email: str
    organization_id: str
    organization_name: str
    api_key: str
    status: str
    profile_pic_url: Optional[str] = None


class DatasetResponse(BaseModel):
    id: str
    date_created: str
    date_updated: Optional[str] = None
    organization_id: str
    data_type: str
    name: str
    description: Optional[str] = None
    status: str
    link: Optional[str] = None
    signed_url: Optional[str] = None
    data_byte_size: Optional[int] = None
    metadata: Optional[dict] = None


class DatasetsListResponse(BaseModel):
    datasets: list[DatasetResponse]
    total: int


class RejectDatasetRequest(BaseModel):
    reason: str


class DatasetStatusUpdateRequest(BaseModel):
    status: str


class StatusUpdateResponse(BaseModel):
    id: str
    status: str
    message: str


class DailyVolumeItem(BaseModel):
    date: str
    count: int


class DatasetStatsResponse(BaseModel):
    total_datasets: int
    total_size_bytes: int
    by_status: dict[str, int]
    by_data_type: dict[str, int]
    daily_volume_last_7_days: list[DailyVolumeItem]
    total_hours_captured: float
    rl_episode_hours_curated: float
    human_synthetic_hours: float


def _one(table: str, column: str, value: str) -> Optional[dict]:
    response = supabase.table(table).select("*").eq(column, value).execute()
    data = getattr(response, "data", None) or []
    return data[0] if data else None


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _verify_password(plain_password: str, hashed_password: str) -> bool:
    return _hash_password(plain_password) == hashed_password


async def verify_api_key(gerra_api_key: str | None = Header(None)) -> dict:
    if not gerra_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Gerra-Api-Key header")

    credential = _one("credential", "api_key", gerra_api_key)
    if credential is None or credential.get("status") != "active":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or inactive API key")

    user = _one("user", "id", credential["user_id"])
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return {"user_id": user["id"], "organization_id": user["organization_id"]}


def _parse_metadata_filters(metadata: Optional[list[str]]) -> Optional[dict[str, str]]:
    if not metadata:
        return None
    filters = {}
    for item in metadata:
        if ":" in item:
            key, value = item.split(":", 1)
            filters[key.strip()] = value.strip()
    return filters or None


def _apply_dataset_filters(
    query,
    organization_id: str,
    data_type: Optional[str],
    status_filter: Optional[str],
    search: Optional[str],
    metadata_filters: Optional[dict[str, str]],
):
    query = query.eq("organization_id", organization_id).neq("status", "for_admin_review")

    if data_type:
        query = query.eq("data_type", data_type)
    if status_filter:
        query = query.eq("status", status_filter)
    if metadata_filters:
        for key, value in metadata_filters.items():
            query = query.contains("metadata", {key: value})

    if search:
        escaped = search.replace("%", r"\%").replace("_", r"\_")
        pattern = f"%{escaped}%"
        query = query.or_(
            ",".join(
                [
                    f"name.ilike.{pattern}",
                    f"description.ilike.{pattern}",
                    f"metadata->task->>description.ilike.{pattern}",
                    f"metadata->task->>instruction.ilike.{pattern}",
                    f"metadata->task->>type.ilike.{pattern}",
                    f"metadata->task->>name.ilike.{pattern}",
                    f"metadata->>episode_id.ilike.{pattern}",
                    f"metadata->>dataset_id.ilike.{pattern}",
                    f"metadata->activity->>name.ilike.{pattern}",
                    f"metadata->activity->>description.ilike.{pattern}",
                    f"metadata->>sample_slug.ilike.{pattern}",
                    f"metadata->>sector.ilike.{pattern}",
                    f"metadata->>flagship_outcome.ilike.{pattern}",
                    f"metadata->>operating_window.ilike.{pattern}",
                ]
            )
        )
    return query


def _storage_path(link: Optional[str]) -> Optional[str]:
    if not link:
        return None
    if link.startswith(("http://", "https://")):
        parsed = urlparse(link)
        parts = parsed.path.split(f"/object/public/{DATASET_BUCKET}/")
        return parts[1] if len(parts) > 1 else None
    if link.startswith(f"{DATASET_BUCKET}/"):
        return link[len(DATASET_BUCKET) + 1 :]
    return link


def _with_signed_url(dataset: dict) -> dict:
    result = dict(dataset)
    result["signed_url"] = None
    path = _storage_path(dataset.get("link"))
    if not path:
        return result
    try:
        signed = supabase.storage.from_(DATASET_BUCKET).create_signed_url(path=path, expires_in=3600)
        result["signed_url"] = signed.get("signedURL") if signed else None
    except Exception as exc:
        print(f"failed to sign dataset URL: {exc}")
    return result


def _fetch_all_datasets(organization_id: str) -> list[dict]:
    rows: list[dict] = []
    page_size = 1000
    offset = 0
    while True:
        response = (
            supabase.table("dataset")
            .select("*")
            .eq("organization_id", organization_id)
            .neq("status", "for_admin_review")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = getattr(response, "data", None) or []
        rows.extend(batch)
        if len(batch) < page_size:
            return rows
        offset += page_size


@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "Gerra Samples API"}


@app.get("/health")
async def health_alias():
    return await health_check()


@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    user = _one("user", "email", request.email)
    if user is None or not _verify_password(request.password, user["password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    credential = _one("credential", "user_id", user["id"])
    if credential is None or credential.get("status") != "active":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account credentials are inactive")

    organization = _one("organization", "id", user["organization_id"])
    if organization is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Organization not found")

    return LoginResponse(
        user_id=user["id"],
        username=user["username"],
        first_name=user["first_name"],
        last_name=user["last_name"],
        email=user["email"],
        organization_id=user["organization_id"],
        organization_name=organization["name"],
        api_key=credential["api_key"],
        status=user.get("status", "active"),
        profile_pic_url=user.get("profile_pic_url"),
    )


@app.get("/datasets", response_model=DatasetsListResponse)
async def list_datasets(
    auth: dict = Depends(verify_api_key),
    data_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    metadata: Optional[list[str]] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    metadata_filters = _parse_metadata_filters(metadata)
    base = supabase.table("dataset").select("*")
    query = _apply_dataset_filters(base, auth["organization_id"], data_type, status, search, metadata_filters)
    response = query.order("date_updated", desc=True).range(offset, offset + limit - 1).execute()
    datasets = getattr(response, "data", None) or []

    count_base = supabase.table("dataset").select("id", count="exact")
    count_query = _apply_dataset_filters(count_base, auth["organization_id"], data_type, status, search, metadata_filters)
    count_response = count_query.execute()
    total = getattr(count_response, "count", None)

    return DatasetsListResponse(datasets=[DatasetResponse(**dataset) for dataset in datasets], total=total or len(datasets))


@app.get("/datasets/dataset", response_model=DatasetResponse)
async def get_dataset(id: str = Query(...), auth: dict = Depends(verify_api_key)):
    dataset = _one("dataset", "id", id)
    if dataset is None or dataset.get("status") == "for_admin_review":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    if dataset["organization_id"] != auth["organization_id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Dataset does not belong to your organization")
    return DatasetResponse(**_with_signed_url(dataset))


@app.get("/datasets/stats", response_model=DatasetStatsResponse)
async def get_dataset_stats(auth: dict = Depends(verify_api_key)):
    datasets = _fetch_all_datasets(auth["organization_id"])
    total_size = sum(dataset.get("data_byte_size", 0) or 0 for dataset in datasets)
    by_status: dict[str, int] = defaultdict(int)
    by_data_type: dict[str, int] = defaultdict(int)
    for dataset in datasets:
        by_status[dataset.get("status", "unknown")] += 1
        by_data_type[dataset.get("data_type", "unknown")] += 1

    today = datetime.now(timezone.utc).date()
    daily_counts = {str(today - timedelta(days=i)): 0 for i in range(7)}
    for dataset in datasets:
        raw = dataset.get("date_created")
        if not raw:
            continue
        try:
            created = datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
        except Exception:
            continue
        key = str(created)
        if key in daily_counts:
            daily_counts[key] += 1

    episode_count = sum(1 for dataset in datasets if dataset.get("data_type") == "episode")
    total_hours = len(datasets) * 0.11
    rl_hours = episode_count * 0.04
    synthetic_hours = total_hours * 0.82 - rl_hours

    return DatasetStatsResponse(
        total_datasets=len(datasets),
        total_size_bytes=total_size,
        by_status=dict(by_status),
        by_data_type=dict(by_data_type),
        daily_volume_last_7_days=[DailyVolumeItem(date=date, count=count) for date, count in sorted(daily_counts.items())],
        total_hours_captured=round(rl_hours + synthetic_hours, 2),
        rl_episode_hours_curated=round(rl_hours, 2),
        human_synthetic_hours=round(synthetic_hours, 2),
    )


@app.post("/datasets/{dataset_id}/accept", response_model=StatusUpdateResponse)
async def accept_dataset(dataset_id: str, auth: dict = Depends(verify_api_key)):
    dataset = _one("dataset", "id", dataset_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    if dataset["organization_id"] != auth["organization_id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Dataset does not belong to your organization")
    supabase.table("dataset").update({"status": "accepted"}).eq("id", dataset_id).execute()
    return StatusUpdateResponse(id=dataset_id, status="accepted", message="Dataset accepted successfully")


@app.patch("/datasets/{dataset_id}/status", response_model=StatusUpdateResponse)
async def update_dataset_status(dataset_id: str, request: DatasetStatusUpdateRequest, auth: dict = Depends(verify_api_key)):
    allowed_statuses = {"accepted", "pending", "rejected"}
    if request.status not in allowed_statuses:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported dataset status")

    dataset = _one("dataset", "id", dataset_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    if dataset["organization_id"] != auth["organization_id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Dataset does not belong to your organization")

    supabase.table("dataset").update({"status": request.status}).eq("id", dataset_id).execute()
    return StatusUpdateResponse(id=dataset_id, status=request.status, message=f"Dataset status updated to {request.status}")


@app.post("/datasets/{dataset_id}/reject", response_model=StatusUpdateResponse)
async def reject_dataset(dataset_id: str, request: RejectDatasetRequest, auth: dict = Depends(verify_api_key)):
    dataset = _one("dataset", "id", dataset_id)
    if dataset is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    if dataset["organization_id"] != auth["organization_id"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Dataset does not belong to your organization")
    metadata = dataset.get("metadata") or {}
    metadata["rejection_reason"] = request.reason
    supabase.table("dataset").update({"status": "rejected", "metadata": metadata}).eq("id", dataset_id).execute()
    return StatusUpdateResponse(id=dataset_id, status="rejected", message=f"Dataset rejected: {request.reason}")
