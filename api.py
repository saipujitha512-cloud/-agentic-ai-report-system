from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from report_system.graph import build_graph


app = FastAPI(
    title="Agentic AI Report System",
    description="Agentic AI API for file analysis and report generation",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Agentic AI Report System API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    goal: str = Form("Analyze the uploaded file and summarize the important findings."),
    title: str = Form("Agentic AI Report"),
    output_format: str = Form("json"),
):
    try:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / file.filename
            output_path = Path(temp_dir) / "report"

            content = await file.read()
            temp_path.write_bytes(content)

            result = build_graph().invoke(
                {
                    "input_path": str(temp_path),
                    "output_path": str(output_path),
                    "output_format": output_format,
                    "title": title,
                    "user_goal": goal,
                }
            )

            return JSONResponse(
                content={
                    "status": "success",
                    "filename": file.filename,
                    "goal": goal,
                    "output_format": output_format,
                    "report": result.get("report"),
                    "quality_review": result.get("quality_review"),
                }
            )

    except Exception as error:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(error),
            },
        )