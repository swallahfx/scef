"""
Routes for the SCEF web interface.
"""

import os
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# Templates
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory="templates")


# Define routes
def register_routes(app):
    @app.get("/ml", response_class=HTMLResponse)
    async def ml_home(request: Request):
        return templates.TemplateResponse("ml/ml_index.html", {"request": request})
    
    @app.get("/ml/models", response_class=HTMLResponse)
    async def ml_models(request: Request):
        return templates.TemplateResponse("ml/ml_models.html", {"request": request})
    
    @app.get("/ml/strategies", response_class=HTMLResponse)
    async def ml_strategies(request: Request):
        return templates.TemplateResponse("ml/ml_strategies.html", {"request": request})
    
    @app.get("/ml/features", response_class=HTMLResponse)
    async def ml_features(request: Request):
        return templates.TemplateResponse("ml/ml_features.html", {"request": request})
