from pydantic import BaseModel, Field

class InputData(BaseModel):
    cpu_usage: float = Field(ge=0.0, le=100.0, description='CPU Usage')
    memory_usage: float = Field(ge=0.0, le=100.0, description='Memory usage')