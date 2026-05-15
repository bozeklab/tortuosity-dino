from typing import Optional

from pydantic import BaseModel


class CMMetrics(BaseModel):
    total: Optional[int] = None
    tn: Optional[int] = None
    tp: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None
