from fastapi import FastAPI, APIRouter, HTTPException, Depends, status, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import jwt
import bcrypt
import base64
from contextlib import asynccontextmanager

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# MongoDB connection
mongo_url = os.environ["MONGO_URL"]

client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

# JWT Configuration
SECRET_KEY = os.environ.get("JWT_SECRET", "ai-pune-secret-key-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

security = HTTPBearer()

# Create the main app
app = FastAPI(title="AI Pune Community API")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# =============================================================================
# MODELS
# =============================================================================


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str
    role: str = "editor"


class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    name: str
    role: str = "editor"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: User


class Event(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    date: str
    time: str
    venue: str
    is_online: bool = False
    description: str
    register_link: str
    image_url: Optional[str] = None
    is_featured: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EventCreate(BaseModel):
    title: str
    date: str
    time: str
    venue: str
    is_online: bool = False
    description: str
    register_link: str
    image_url: Optional[str] = None
    is_featured: bool = False


class Hackathon(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    date: str
    end_date: Optional[str] = None
    venue: str
    description: str
    register_link: str
    image_url: Optional[str] = None
    status: str = "upcoming"  # upcoming, ongoing, completed
    sponsors: List[str] = []
    winners: List[dict] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HackathonCreate(BaseModel):
    title: str
    date: str
    end_date: Optional[str] = None
    venue: str
    description: str
    register_link: str
    image_url: Optional[str] = None
    status: str = "upcoming"
    sponsors: List[str] = []
    winners: List[dict] = []


class Speaker(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    designation: str
    company: str
    photo_url: Optional[str] = None
    linkedin: Optional[str] = None
    twitter: Optional[str] = None
    sessions: List[str] = []
    is_featured: bool = False
    order: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SpeakerCreate(BaseModel):
    name: str
    designation: str
    company: str
    photo_url: Optional[str] = None
    linkedin: Optional[str] = None
    twitter: Optional[str] = None
    sessions: List[str] = []
    is_featured: bool = False
    order: int = 0


class Organizer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    role: str  # Lead Organizer, Co-Organizer, Volunteer
    bio: str
    photo_url: Optional[str] = None
    linkedin: Optional[str] = None
    order: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OrganizerCreate(BaseModel):
    name: str
    role: str
    bio: str
    photo_url: Optional[str] = None
    linkedin: Optional[str] = None
    order: int = 0


class Partner(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    logo_url: str
    website: Optional[str] = None
    order: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PartnerCreate(BaseModel):
    name: str
    logo_url: str
    website: Optional[str] = None
    order: int = 0


class SiteContent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = "main"
    # Hero Section
    hero_tagline: str = "Building the Future of AI Together"
    hero_subtext: str = "Backed by Google for Developers"
    hero_image: str = (
        "https://images.unsplash.com/photo-1664526936810-ec0856d31b92?w=1920&q=80"
    )
    hero_cta_primary_text: str = "Upcoming Events"
    hero_cta_primary_link: str = "/events"
    hero_cta_secondary_text: str = "Join Community"
    hero_cta_secondary_link: str = "/about"
    hero_active: bool = True
    # Home Page Sections Active Status
    about_section_active: bool = True
    events_section_active: bool = True
    hackathons_section_active: bool = True
    speakers_section_active: bool = True
    organizers_section_active: bool = True
    partners_section_active: bool = True
    cta_section_active: bool = True
    # CTA Section
    cta_title: str = "Ready to Join the AI Revolution?"
    cta_subtitle: str = (
        "Be part of Pune's most vibrant AI community. Attend events, participate in hackathons, and connect with like-minded individuals."
    )
    cta_button_text: str = "Get Started Today"
    cta_button_link: str = "/events"
    # About Page Content
    about_intro: str = ""
    about_who_we_are: str = ""
    about_vision: str = ""
    about_mission: str = ""
    about_history: str = ""
    about_hero_image: str = (
        "https://images.unsplash.com/photo-1531482615713-2afd69097998?w=800&q=80"
    )
    # General
    community_size: str = "3000+"
    footer_text: str = ""
    # Social Links
    social_linkedin: str = "https://linkedin.com/company/aipune"
    social_linkedin_active: bool = True
    social_twitter: str = "https://twitter.com/aipune"
    social_twitter_active: bool = True
    social_youtube: str = "https://youtube.com/@aipune"
    social_youtube_active: bool = True
    social_email: str = "hello@aipune.org"
    social_email_active: bool = True
    social_whatsapp: str = ""
    social_whatsapp_active: bool = False
    # Code of Conduct
    code_of_conduct: dict = {}
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SiteContentUpdate(BaseModel):
    # Hero Section
    hero_tagline: Optional[str] = None
    hero_subtext: Optional[str] = None
    hero_image: Optional[str] = None
    hero_cta_primary_text: Optional[str] = None
    hero_cta_primary_link: Optional[str] = None
    hero_cta_secondary_text: Optional[str] = None
    hero_cta_secondary_link: Optional[str] = None
    hero_active: Optional[bool] = None
    # Home Page Sections
    about_section_active: Optional[bool] = None
    events_section_active: Optional[bool] = None
    hackathons_section_active: Optional[bool] = None
    speakers_section_active: Optional[bool] = None
    organizers_section_active: Optional[bool] = None
    partners_section_active: Optional[bool] = None
    cta_section_active: Optional[bool] = None
    # CTA Section
    cta_title: Optional[str] = None
    cta_subtitle: Optional[str] = None
    cta_button_text: Optional[str] = None
    cta_button_link: Optional[str] = None
    # About Page
    about_intro: Optional[str] = None
    about_who_we_are: Optional[str] = None
    about_vision: Optional[str] = None
    about_mission: Optional[str] = None
    about_history: Optional[str] = None
    about_hero_image: Optional[str] = None
    # General
    community_size: Optional[str] = None
    footer_text: Optional[str] = None
    # Social Links
    social_linkedin: Optional[str] = None
    social_linkedin_active: Optional[bool] = None
    social_twitter: Optional[str] = None
    social_twitter_active: Optional[bool] = None
    social_youtube: Optional[str] = None
    social_youtube_active: Optional[bool] = None
    social_email: Optional[str] = None
    social_email_active: Optional[bool] = None
    social_whatsapp: Optional[str] = None
    social_whatsapp_active: Optional[bool] = None
    # Code of Conduct
    code_of_conduct: Optional[dict] = None


class Media(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    url: str
    content_type: str
    size: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# AUTH HELPERS
# =============================================================================


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    try:
        payload = jwt.decode(
            credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM]
        )
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = await db.users.find_one({"id": user_id}, {"_id": 0, "password_hash": 0})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def require_admin(user: dict = Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# =============================================================================
# AUTH ROUTES
# =============================================================================


@api_router.post("/auth/login", response_model=TokenResponse)
async def login(data: UserLogin):
    user = await db.users.find_one({"email": data.email}, {"_id": 0})
    if not user or not verify_password(data.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user["id"], "role": user["role"]})
    user_data = {k: v for k, v in user.items() if k != "password_hash"}
    return TokenResponse(access_token=token, user=User(**user_data))


@api_router.post("/auth/register", response_model=TokenResponse)
async def register(data: UserRegister, current_user: dict = Depends(require_admin)):
    existing = await db.users.find_one({"email": data.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(email=data.email, name=data.name, role=data.role)
    user_dict = user.model_dump()
    user_dict["password_hash"] = hash_password(data.password)
    user_dict["created_at"] = user_dict["created_at"].isoformat()

    await db.users.insert_one(user_dict)
    token = create_access_token({"sub": user.id, "role": user.role})
    return TokenResponse(access_token=token, user=user)


@api_router.get("/auth/me", response_model=User)
async def get_me(current_user: dict = Depends(get_current_user)):
    return User(**current_user)


# =============================================================================
# EVENTS ROUTES
# =============================================================================


@api_router.get("/events", response_model=List[Event])
async def get_events():
    events = await db.events.find({}, {"_id": 0}).sort("date", -1).to_list(100)
    return events


@api_router.get("/events/featured", response_model=List[Event])
async def get_featured_events():
    events = (
        await db.events.find({"is_featured": True}, {"_id": 0})
        .sort("date", -1)
        .to_list(10)
    )
    return events


@api_router.get("/events/{event_id}", response_model=Event)
async def get_event(event_id: str):
    event = await db.events.find_one({"id": event_id}, {"_id": 0})
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


@api_router.post("/events", response_model=Event)
async def create_event(
    data: EventCreate, current_user: dict = Depends(get_current_user)
):
    event = Event(**data.model_dump())
    doc = event.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.events.insert_one(doc)
    return event


@api_router.put("/events/{event_id}", response_model=Event)
async def update_event(
    event_id: str, data: EventCreate, current_user: dict = Depends(get_current_user)
):
    result = await db.events.update_one({"id": event_id}, {"$set": data.model_dump()})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Event not found")
    event = await db.events.find_one({"id": event_id}, {"_id": 0})
    return event


@api_router.delete("/events/{event_id}")
async def delete_event(event_id: str, current_user: dict = Depends(get_current_user)):
    result = await db.events.delete_one({"id": event_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Event not found")
    return {"message": "Event deleted"}


# =============================================================================
# HACKATHONS ROUTES
# =============================================================================


@api_router.get("/hackathons", response_model=List[Hackathon])
async def get_hackathons():
    hackathons = await db.hackathons.find({}, {"_id": 0}).sort("date", -1).to_list(100)
    return hackathons


@api_router.get("/hackathons/status/{status}", response_model=List[Hackathon])
async def get_hackathons_by_status(status: str):
    hackathons = (
        await db.hackathons.find({"status": status}, {"_id": 0})
        .sort("date", -1)
        .to_list(100)
    )
    return hackathons


@api_router.get("/hackathons/{hackathon_id}", response_model=Hackathon)
async def get_hackathon(hackathon_id: str):
    hackathon = await db.hackathons.find_one({"id": hackathon_id}, {"_id": 0})
    if not hackathon:
        raise HTTPException(status_code=404, detail="Hackathon not found")
    return hackathon


@api_router.post("/hackathons", response_model=Hackathon)
async def create_hackathon(
    data: HackathonCreate, current_user: dict = Depends(get_current_user)
):
    hackathon = Hackathon(**data.model_dump())
    doc = hackathon.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.hackathons.insert_one(doc)
    return hackathon


@api_router.put("/hackathons/{hackathon_id}", response_model=Hackathon)
async def update_hackathon(
    hackathon_id: str,
    data: HackathonCreate,
    current_user: dict = Depends(get_current_user),
):
    result = await db.hackathons.update_one(
        {"id": hackathon_id}, {"$set": data.model_dump()}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Hackathon not found")
    hackathon = await db.hackathons.find_one({"id": hackathon_id}, {"_id": 0})
    return hackathon


@api_router.delete("/hackathons/{hackathon_id}")
async def delete_hackathon(
    hackathon_id: str, current_user: dict = Depends(get_current_user)
):
    result = await db.hackathons.delete_one({"id": hackathon_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Hackathon not found")
    return {"message": "Hackathon deleted"}


# =============================================================================
# SPEAKERS ROUTES
# =============================================================================


@api_router.get("/speakers", response_model=List[Speaker])
async def get_speakers():
    speakers = await db.speakers.find({}, {"_id": 0}).sort("order", 1).to_list(100)
    return speakers


@api_router.get("/speakers/featured", response_model=List[Speaker])
async def get_featured_speakers():
    speakers = (
        await db.speakers.find({"is_featured": True}, {"_id": 0})
        .sort("order", 1)
        .to_list(10)
    )
    return speakers


@api_router.get("/speakers/{speaker_id}", response_model=Speaker)
async def get_speaker(speaker_id: str):
    speaker = await db.speakers.find_one({"id": speaker_id}, {"_id": 0})
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")
    return speaker


@api_router.post("/speakers", response_model=Speaker)
async def create_speaker(
    data: SpeakerCreate, current_user: dict = Depends(get_current_user)
):
    speaker = Speaker(**data.model_dump())
    doc = speaker.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.speakers.insert_one(doc)
    return speaker


@api_router.put("/speakers/{speaker_id}", response_model=Speaker)
async def update_speaker(
    speaker_id: str, data: SpeakerCreate, current_user: dict = Depends(get_current_user)
):
    result = await db.speakers.update_one(
        {"id": speaker_id}, {"$set": data.model_dump()}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Speaker not found")
    speaker = await db.speakers.find_one({"id": speaker_id}, {"_id": 0})
    return speaker


@api_router.delete("/speakers/{speaker_id}")
async def delete_speaker(
    speaker_id: str, current_user: dict = Depends(get_current_user)
):
    result = await db.speakers.delete_one({"id": speaker_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Speaker not found")
    return {"message": "Speaker deleted"}


# =============================================================================
# ORGANIZERS ROUTES
# =============================================================================


@api_router.get("/organizers", response_model=List[Organizer])
async def get_organizers():
    organizers = await db.organizers.find({}, {"_id": 0}).sort("order", 1).to_list(100)
    return organizers


@api_router.get("/organizers/{organizer_id}", response_model=Organizer)
async def get_organizer(organizer_id: str):
    organizer = await db.organizers.find_one({"id": organizer_id}, {"_id": 0})
    if not organizer:
        raise HTTPException(status_code=404, detail="Organizer not found")
    return organizer


@api_router.post("/organizers", response_model=Organizer)
async def create_organizer(
    data: OrganizerCreate, current_user: dict = Depends(get_current_user)
):
    organizer = Organizer(**data.model_dump())
    doc = organizer.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.organizers.insert_one(doc)
    return organizer


@api_router.put("/organizers/{organizer_id}", response_model=Organizer)
async def update_organizer(
    organizer_id: str,
    data: OrganizerCreate,
    current_user: dict = Depends(get_current_user),
):
    result = await db.organizers.update_one(
        {"id": organizer_id}, {"$set": data.model_dump()}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Organizer not found")
    organizer = await db.organizers.find_one({"id": organizer_id}, {"_id": 0})
    return organizer


@api_router.delete("/organizers/{organizer_id}")
async def delete_organizer(
    organizer_id: str, current_user: dict = Depends(get_current_user)
):
    result = await db.organizers.delete_one({"id": organizer_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Organizer not found")
    return {"message": "Organizer deleted"}


# =============================================================================
# PARTNERS ROUTES
# =============================================================================


@api_router.get("/partners", response_model=List[Partner])
async def get_partners():
    partners = await db.partners.find({}, {"_id": 0}).sort("order", 1).to_list(100)
    return partners


@api_router.post("/partners", response_model=Partner)
async def create_partner(
    data: PartnerCreate, current_user: dict = Depends(get_current_user)
):
    partner = Partner(**data.model_dump())
    doc = partner.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.partners.insert_one(doc)
    return partner


@api_router.put("/partners/{partner_id}", response_model=Partner)
async def update_partner(
    partner_id: str, data: PartnerCreate, current_user: dict = Depends(get_current_user)
):
    result = await db.partners.update_one(
        {"id": partner_id}, {"$set": data.model_dump()}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Partner not found")
    partner = await db.partners.find_one({"id": partner_id}, {"_id": 0})
    return partner


@api_router.delete("/partners/{partner_id}")
async def delete_partner(
    partner_id: str, current_user: dict = Depends(get_current_user)
):
    result = await db.partners.delete_one({"id": partner_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Partner not found")
    return {"message": "Partner deleted"}


# =============================================================================
# SITE CONTENT ROUTES
# =============================================================================


@api_router.get("/content", response_model=SiteContent)
async def get_site_content():
    content = await db.site_content.find_one({"id": "main"}, {"_id": 0})
    if not content:
        # Return default content
        return SiteContent()
    return content


@api_router.put("/content", response_model=SiteContent)
async def update_site_content(
    data: SiteContentUpdate, current_user: dict = Depends(get_current_user)
):
    update_data = {k: v for k, v in data.model_dump().items() if v is not None}
    update_data["updated_at"] = datetime.now(timezone.utc).isoformat()

    await db.site_content.update_one({"id": "main"}, {"$set": update_data}, upsert=True)
    content = await db.site_content.find_one({"id": "main"}, {"_id": 0})
    return content


# =============================================================================
# MEDIA ROUTES
# =============================================================================


@api_router.get("/media", response_model=List[Media])
async def get_media(current_user: dict = Depends(get_current_user)):
    media = await db.media.find({}, {"_id": 0}).sort("created_at", -1).to_list(100)
    return media


@api_router.post("/media", response_model=Media)
async def upload_media(
    file: UploadFile = File(...), current_user: dict = Depends(get_current_user)
):
    content = await file.read()
    encoded = base64.b64encode(content).decode("utf-8")

    media = Media(
        filename=file.filename,
        url=f"data:{file.content_type};base64,{encoded}",
        content_type=file.content_type,
        size=len(content),
    )
    doc = media.model_dump()
    doc["created_at"] = doc["created_at"].isoformat()
    await db.media.insert_one(doc)
    return media


@api_router.delete("/media/{media_id}")
async def delete_media(media_id: str, current_user: dict = Depends(get_current_user)):
    result = await db.media.delete_one({"id": media_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Media not found")
    return {"message": "Media deleted"}


# =============================================================================
# SEED DATA ROUTE
# =============================================================================


@api_router.post("/seed")
async def seed_database():
    # Check if already seeded
    admin = await db.users.find_one({"email": "admin@aipune.org"})
    if admin:
        return {"message": "Database already seeded"}

    # Create admin user
    admin_user = {
        "id": str(uuid.uuid4()),
        "email": "admin@aipune.org",
        "name": "AI Pune Admin",
        "role": "admin",
        "password_hash": hash_password("AIPune@2024"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.users.insert_one(admin_user)

    # Seed Events
    events = [
        {
            "id": str(uuid.uuid4()),
            "title": "Introduction to Generative AI with Google",
            "date": "2025-02-15",
            "time": "10:00 AM - 4:00 PM",
            "venue": "Persistent Systems, Hinjewadi",
            "is_online": False,
            "description": "Learn the fundamentals of Generative AI, explore Google's AI tools, and build your first AI application in this hands-on workshop.",
            "register_link": "https://commudle.com/aipune",
            "image_url": "https://images.unsplash.com/photo-1677442136019-21780ecad995?w=800",
            "is_featured": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": str(uuid.uuid4()),
            "title": "TensorFlow Developer Summit Watch Party",
            "date": "2025-03-10",
            "time": "6:00 PM - 9:00 PM",
            "venue": "Online - Google Meet",
            "is_online": True,
            "description": "Join us for a live watch party of the TensorFlow Developer Summit with live discussions and networking opportunities.",
            "register_link": "https://commudle.com/aipune",
            "image_url": "https://images.unsplash.com/photo-1591453089816-0fbb971b454c?w=800",
            "is_featured": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": str(uuid.uuid4()),
            "title": "MLOps Workshop: Production-Ready ML Pipelines",
            "date": "2025-03-25",
            "time": "9:00 AM - 5:00 PM",
            "venue": "Tech Park One, Kharadi",
            "is_online": False,
            "description": "Deep dive into MLOps practices, CI/CD for ML, model monitoring, and deployment strategies using Google Cloud.",
            "register_link": "https://commudle.com/aipune",
            "image_url": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800",
            "is_featured": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    ]
    await db.events.insert_many(events)

    # Seed Hackathons
    hackathons = [
        {
            "id": str(uuid.uuid4()),
            "title": "AI for Good Hackathon 2025",
            "date": "2025-04-12",
            "end_date": "2025-04-14",
            "venue": "Symbiosis Institute of Technology, Lavale",
            "description": "48-hour hackathon focused on building AI solutions for social impact. Themes include healthcare, education, environment, and accessibility.",
            "register_link": "https://commudle.com/aipune/hackathons",
            "image_url": "https://images.unsplash.com/photo-1504384308090-c894fdcc538d?w=800",
            "status": "upcoming",
            "sponsors": ["Google", "Persistent Systems", "TCS"],
            "winners": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": str(uuid.uuid4()),
            "title": "GenAI Build Challenge",
            "date": "2024-11-20",
            "end_date": "2024-11-22",
            "venue": "MIT World Peace University",
            "description": "Build innovative applications using Generative AI APIs from Google, OpenAI, and more. Cash prizes worth ₹2 Lakhs!",
            "register_link": "https://commudle.com/aipune/hackathons",
            "image_url": "https://images.unsplash.com/photo-1531482615713-2afd69097998?w=800",
            "status": "completed",
            "sponsors": ["Google", "Nvidia", "Infosys"],
            "winners": [
                {
                    "team": "Neural Ninjas",
                    "position": 1,
                    "project": "AI Health Assistant",
                },
                {
                    "team": "Code Crafters",
                    "position": 2,
                    "project": "Smart Education Platform",
                },
                {"team": "Data Wizards", "position": 3, "project": "Eco Monitor AI"},
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    ]
    await db.hackathons.insert_many(hackathons)

    # Seed Speakers
    speakers = [
        {
            "id": str(uuid.uuid4()),
            "name": "Dr. Priya Sharma",
            "designation": "ML Research Lead",
            "company": "Google India",
            "photo_url": "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=400",
            "linkedin": "https://linkedin.com/in/",
            "twitter": "https://twitter.com/",
            "sessions": ["Intro to TensorFlow", "Advanced NLP Techniques"],
            "is_featured": True,
            "order": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Rahul Mehta",
            "designation": "Senior AI Engineer",
            "company": "Microsoft",
            "photo_url": "https://images.unsplash.com/photo-1560250097-0b93528c311a?w=400",
            "linkedin": "https://linkedin.com/in/",
            "twitter": "https://twitter.com/",
            "sessions": ["Azure ML Workshop", "Computer Vision Applications"],
            "is_featured": True,
            "order": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Ananya Desai",
            "designation": "Data Science Manager",
            "company": "Persistent Systems",
            "photo_url": "https://images.unsplash.com/photo-1580489944761-15a19d654956?w=400",
            "linkedin": "https://linkedin.com/in/",
            "twitter": "https://twitter.com/",
            "sessions": ["MLOps Best Practices", "Data Engineering for ML"],
            "is_featured": True,
            "order": 3,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Vikram Patil",
            "designation": "AI Product Manager",
            "company": "Nvidia",
            "photo_url": "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=400",
            "linkedin": "https://linkedin.com/in/",
            "twitter": "https://twitter.com/",
            "sessions": ["GPU Computing for AI", "Deep Learning Optimization"],
            "is_featured": False,
            "order": 4,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    ]
    await db.speakers.insert_many(speakers)

    # Seed Organizers
    organizers = [
        {
            "id": str(uuid.uuid4()),
            "name": "Amit Kumar",
            "role": "Lead Organizer",
            "bio": "Passionate about building AI communities. 10+ years in tech leadership. Google Developer Expert in ML.",
            "photo_url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",
            "linkedin": "https://linkedin.com/in/",
            "order": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Sneha Kulkarni",
            "role": "Co-Organizer",
            "bio": "Full-stack developer turned AI enthusiast. Organizes workshops and mentors students in ML/AI.",
            "photo_url": "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=400",
            "linkedin": "https://linkedin.com/in/",
            "order": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Rohan Joshi",
            "role": "Co-Organizer",
            "bio": "Data scientist at heart. Loves building ML products and sharing knowledge with the community.",
            "photo_url": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e?w=400",
            "linkedin": "https://linkedin.com/in/",
            "order": 3,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Prachi Shah",
            "role": "Volunteer",
            "bio": "CS student passionate about NLP and computer vision. Active contributor to open-source AI projects.",
            "photo_url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400",
            "linkedin": "https://linkedin.com/in/",
            "order": 4,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    ]
    await db.organizers.insert_many(organizers)

    # Seed Partners
    partners = [
        {
            "id": str(uuid.uuid4()),
            "name": "Google for Developers",
            "logo_url": "https://www.gstatic.com/devrel-devsite/prod/v0e0f589edd85502a40d78d7d0825db8ea5ef3b99ab4070381ee86977c9168730/developers/images/lockup.svg",
            "website": "https://developers.google.com/",
            "order": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Persistent Systems",
            "logo_url": "https://www.persistent.com/wp-content/uploads/2020/05/persistent-logo.svg",
            "website": "https://www.persistent.com/",
            "order": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": str(uuid.uuid4()),
            "name": "TCS",
            "logo_url": "https://www.tcs.com/content/dam/tcs/images/og-images/TCS_OG_Image.png",
            "website": "https://www.tcs.com/",
            "order": 3,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    ]
    await db.partners.insert_many(partners)

    # Seed Site Content
    site_content = {
        "id": "main",
        # Hero Section
        "hero_tagline": "Building the Future of AI Together",
        "hero_subtext": "Backed by Google for Developers",
        "hero_image": "https://images.unsplash.com/photo-1664526936810-ec0856d31b92?w=1920&q=80",
        "hero_cta_primary_text": "Upcoming Events",
        "hero_cta_primary_link": "/events",
        "hero_cta_secondary_text": "Join Community",
        "hero_cta_secondary_link": "/about",
        "hero_active": True,
        # Home Page Sections
        "about_section_active": True,
        "events_section_active": True,
        "hackathons_section_active": True,
        "speakers_section_active": True,
        "organizers_section_active": True,
        "partners_section_active": True,
        "cta_section_active": True,
        # CTA Section
        "cta_title": "Ready to Join the AI Revolution?",
        "cta_subtitle": "Be part of Pune's most vibrant AI community. Attend events, participate in hackathons, and connect with like-minded individuals.",
        "cta_button_text": "Get Started Today",
        "cta_button_link": "/events",
        # About Page
        "about_intro": "AI Pune is a vibrant community of AI and Machine Learning enthusiasts, developers, researchers, and students based in Pune, India. Backed by Google for Developers, we are dedicated to fostering knowledge sharing, collaboration, and innovation in the field of artificial intelligence.",
        "about_who_we_are": "We are a diverse group of professionals, students, and enthusiasts united by our passion for AI and machine learning. Our community includes data scientists, ML engineers, researchers, students, and anyone curious about the future of AI.",
        "about_vision": "To be the leading AI community in India, driving innovation and creating opportunities for learning and collaboration in artificial intelligence.",
        "about_mission": "To democratize AI education, foster a collaborative environment for AI practitioners, and build real-world AI solutions that benefit society.",
        "about_history": "Founded as TFUG Pune (TensorFlow User Group Pune) in 2018, we have grown into one of the largest AI communities in Maharashtra. In 2024, we rebranded to AI Pune to better reflect our expanded focus on all aspects of artificial intelligence.",
        "about_hero_image": "https://images.unsplash.com/photo-1531482615713-2afd69097998?w=800&q=80",
        # General
        "community_size": "3000+",
        "footer_text": "AI Pune (Previously Known as TFUG Pune) | Backed by Google for Developers",
        # Social Links
        "social_linkedin": "https://linkedin.com/company/aipune",
        "social_twitter": "https://twitter.com/aipune",
        "social_youtube": "https://youtube.com/@aipune",
        "social_email": "hello@aipune.org",
        "social_whatsapp": "",
        "social_whatsapp_active": False,
        # Code of Conduct
        "code_of_conduct": {
            "commitment": "AI Pune is dedicated to providing a harassment-free experience for everyone, regardless of gender, gender identity and expression, age, sexual orientation, disability, physical appearance, body size, race, ethnicity, religion, or technology choices.",
            "expected_behavior": [
                "Be respectful and inclusive in your speech and actions",
                "Refrain from demeaning, discriminatory, or harassing behavior",
                "Be mindful of your surroundings and fellow participants",
                "Alert community leaders if you notice a dangerous situation or someone in distress",
            ],
            "unacceptable_behavior": [
                "Harassment, intimidation, or discrimination in any form",
                "Physical, verbal, or written abuse",
                "Unwelcome sexual attention or advances",
                "Deliberate intimidation, stalking, or following",
                "Disruption of talks or other events",
            ],
            "reporting": "If you experience or witness unacceptable behavior, please report it immediately to any of the community organizers. All reports will be handled with discretion.",
            "enforcement": "Community organizers will take appropriate action in response to any violation of this Code of Conduct, which may include warning the offender, expulsion from the event, or banning from future events.",
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.site_content.insert_one(site_content)

    return {
        "message": "Database seeded successfully",
        "admin_email": "admin@aipune.org",
        "admin_password": "AIPune@2024",
    }


# =============================================================================
# HEALTH CHECK
# =============================================================================


@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("START")
    yield
    print("STOP")


# @app.on_event("shutdown")
# async def shutdown_db_client():
#     client.close()
