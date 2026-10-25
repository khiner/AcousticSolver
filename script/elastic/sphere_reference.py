#!/usr/bin/env python3
"""Compare the coupled DG operator with e3Dss's independent elastic-sphere solution."""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def octave_reference(output):
    # Octave equivalents for MATLAB's pool query and field-name alias; numerical code is unchanged.
    (output / "gcp.m").write_text("function pool=gcp(varargin)\npool=[];\nend\n")
    (output / "fields.m").write_text("function names=fields(value)\nnames=fieldnames(value);\nend\n")
    quote = lambda p: str(p).replace("'", "''")
    source = f"""function SphereReference
addpath('{quote(ROOT/'external/e3dss')}');addpath('{quote(ROOT/'external/e3dss/utils')}');
data=jsondecode(fileread('points.json'));omega=2;
result.omega=omega;result.max_terms=0;
for group={{'elements','boundaries'}}
 name=group{{1}};items=data.(name);result.(name)=cell(numel(items),1);
 for k=1:numel(items)
  x=items(k).points;tag=items(k).tag;
  layer={{}};
  layer{{1}}=struct('media','fluid','R',.4,'rho',1,'c_f',1,'X',[0,0,.8], 'calc_p',true,'calc_p_inc',true,'calc_dp',true(1,3),'calc_dp_inc',true(1,3));
  layer{{2}}=struct('media','solid','R',0,'rho',2,'E',5,'nu',.25,'X',[0,0,.2], 'calc_u',true(1,3),'calc_sigma',true(1,6));
  layer{{tag}}.X=x;
  options=struct('omega',omega,'BC','NNBC','Display','none','nu_a',-1,'N_max',30);
  [layer,terms,flag]=e3Dss(layer,options);
  if any(flag);error('Sphere series failed to converge');end
  result.max_terms=max(result.max_terms,max(terms));
  if tag==1
   q=zeros(size(x,1),4);q(:,4)=-(layer{{1}}.p(:)+layer{{1}}.p_inc(:));
   for d=1:3;q(:,d)=(layer{{1}}.dp{{d}}(:)+layer{{1}}.dp_inc{{d}}(:))/(1i*omega);end
  else
   q=zeros(size(x,1),9);
   for d=1:3;q(:,d)=-1i*omega*layer{{2}}.u{{d}}(:);end
   for d=1:6;q(:,d+3)=layer{{2}}.sigma{{d}}(:);end
  end
  result.(name){{k}}=struct('real',real(q),'imag',imag(q));
 end
end
fid=fopen('fields.json','w');fputs(fid,jsonencode(result));fclose(fid);
end
"""
    (output / "SphereReference.m").write_text(source)
    with (output / "octave.log").open("w") as log:
        subprocess.run(
            ["octave", "--quiet", "--no-gui", "--eval", "SphereReference;"],
            cwd=output,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return hashlib.sha256(source.encode()).hexdigest()


def run(output, spacing, degrees):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runpy.run_path(str(ROOT / "script/ElasticDemo"))["mesh"](output, spacing, optimize=False)
    reports = []
    for degree in degrees:
        case = output / f"p{degree}"
        case.mkdir(exist_ok=True)
        config = {
            "mesh": "../mesh.json",
            "degree": degree,
            "materials": {
                "1": {"type": "fluid", "density": 1, "sound_speed": 1},
                "2": {"type": "solid", "density": 2, "young_modulus": 5, "poisson_ratio": 0.25},
            },
        }
        (case / "config.json").write_text(json.dumps(config))
        command = [
            str(ROOT / "build/ElasticReference"),
            str(case / "config.json"),
            str(case / "points.json"),
            str(case / "report.json"),
        ]
        subprocess.run(command, check=True)
        harness_hash = octave_reference(case)
        subprocess.run(command + [str(case / "fields.json")], check=True)
        reports.append(json.loads((case / "report.json").read_text()))
        reports[-1]["harness_sha256"] = harness_hash
    report = {
        "upstream": subprocess.check_output(
            ["git", "-C", str(ROOT / "external/e3dss"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "mesh_sha256": hashlib.sha256((output / "mesh.json").read_bytes()).hexdigest(),
        "results": reports,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for tag in ("1", "2"):
        errors = [r["relative_errors"][tag] for r in reports]
        if errors[-1] > 0.03 or (len(errors) > 1 and errors[-1] > 0.8 * errors[0]):
            raise RuntimeError(f"Sphere convergence/accuracy failed for region {tag}: {errors}")
    print(output / "report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/elastic-reference/sphere")
    parser.add_argument("--spacing", type=float, default=0.7)
    parser.add_argument("--degrees", nargs="+", type=int, default=[1, 2])
    args = parser.parse_args()
    run(args.output, args.spacing, args.degrees)
