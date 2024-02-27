from pydantic import BaseModel, conlist
from typing import List

class Iris(BaseModel):
    data_input: List[conlist(float, min_length=4, max_length=4)]
    