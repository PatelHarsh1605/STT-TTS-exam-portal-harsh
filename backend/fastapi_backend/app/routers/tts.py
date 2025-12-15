from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.schemas.tts import TTSRequest
from app.services.tts_service import generate_tts_audio

router = APIRouter(prefix="/tts", tags=["tts"])

@router.post("/synthesize")
async def synthesize(payload: TTSRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    audio_path = generate_tts_audio(
        text=payload.text,
        language=payload.language,
        slow=payload.slow
    )

    return FileResponse(
        path=audio_path,
        media_type="audio/mpeg",
        filename="tts.mp3"
    )
