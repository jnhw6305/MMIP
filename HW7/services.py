"""Downstream dispatch, retry, circuit-breaker style fallback, and audit evidence."""
from __future__ import annotations
import time, uuid
from dataclasses import dataclass, asdict
from typing import Any, Callable

AUDIT_LOG: list[dict[str, Any]]=[]

@dataclass
class DispatchResult:
    ticket_id: str
    destination: str
    status: str
    attempts: int
    fallback: bool
    error: str | None=None

def dispatch(route: str, payload: dict[str, Any], handler: Callable | None=None, max_attempts: int=3) -> DispatchResult:
    ticket=f"T-{uuid.uuid4().hex[:8].upper()}";destination=route if route not in {"other","security"} else ("human" if route=="other" else "security")
    attempts=0;error=None
    for attempts in range(1,max_attempts+1):
        try:
            response=handler(payload) if handler else {"accepted":True}
            if not response.get("accepted"): raise RuntimeError("downstream rejected request")
            result=DispatchResult(ticket,destination,"queued",attempts,False)
            break
        except Exception as exc:
            error=str(exc)
            if attempts<max_attempts: time.sleep(.01*(2**(attempts-1)))
    else:
        result=DispatchResult(ticket,"human","fallback",attempts,True,error)
    AUDIT_LOG.append({"event":"dispatch","route":route,"payload_keys":sorted(payload),**asdict(result)})
    return result
