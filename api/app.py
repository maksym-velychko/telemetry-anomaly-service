from fastapi import FastAPI

from api.routes import router_v1

app = FastAPI(title='Telemetry data service 🤖')
app.include_router(router_v1)