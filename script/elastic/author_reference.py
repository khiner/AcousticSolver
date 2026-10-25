#!/usr/bin/env python3
"""Run the pinned authors' coupled MATLAB solver headlessly in Octave."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "external/wadg-elastic"


def run(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = UPSTREAM / "ElasticityAcoustic2D_unified.m"
    text = source.read_text()
    start = text.index("function [rhs] = ElasRHS2D")
    body, functions = text[:start], text[start:]
    body = body[: body.index("% keyboard\nset(gca")]
    body = body.replace("clear all, clear", "")
    body = (
        body.replace("K1D = 16;", "K1D = 4;")
        .replace("N = 4;", "N = 3;")
        .replace("FinalTime = 2.0;", "FinalTime = .02;")
    )
    body = body.replace("if Nfld*Np*K < 2500", "if false")
    body = body.replace("if 1 && (mod(tstep,10)==0 || tstep==Nsteps)", "if false")
    # Match stage clocks to the state being advanced; upstream starts boundary forcing one step ahead.
    body = body.replace("time = tstep*dt;", "time = (tstep-1)*dt;")
    capture = """
result.x=x(:); result.y=y(:); result.time=FinalTime; result.dt=dt;
result.degree=N; result.elements=K;
result.state=zeros(numel(x),5);
for field=1:5; result.state(:,field)=U{field}(:); end
result.exact=zeros(numel(x),5);
result.exact(:,1)=reshape(u1a(x,y,FinalTime),[],1);
result.exact(:,2)=reshape(u2a(x,y,FinalTime),[],1);
result.exact(:,3)=reshape(s1ax(x,y,FinalTime),[],1);
for field=1:5
 if field==1; values=u1e(x,y,FinalTime); elseif field==2; values=u2e(x,y,FinalTime);
 elseif field==3; values=s1ex(x,y,FinalTime); elseif field==4; values=s2ey(x,y,FinalTime);
 else; values=s12exy(x,y,FinalTime); end
 mask=zeros(size(x));mask(:,Ke)=1;selected=find(mask(:));result.exact(selected,field)=values(selected);
end
fid=fopen('author.json','w');fputs(fid,jsonencode(result));fclose(fid);
end
"""
    adapted = output / "RunAuthor.m"
    adapted.write_text("function RunAuthor\n" + body + capture + functions)
    quote = lambda p: str(p).replace("'", "''")
    command = f"addpath('{quote(UPSTREAM)}'); addpath('{quote(output)}'); cd('{quote(output)}'); RunAuthor;"
    with (output / "octave.log").open("w") as log:
        subprocess.run(
            ["octave", "--no-gui", "--quiet", "--eval", command],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    manifest = {
        "revision": subprocess.check_output(
            ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(adapted.read_bytes()).hexdigest(),
        "changes": [
            "function wrapper and headless output",
            "bounded degree/mesh/duration",
            "align boundary stage clock to beginning of timestep",
        ],
    }
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output / "author.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/elastic-reference/author")
    run(parser.parse_args().output)
