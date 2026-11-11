import csv,hashlib,json,pathlib,subprocess
p=pathlib.Path.home()/"dev/thesis/models/WFM-InformedSource/runs/shared32-batch4-20261107"
plan=json.loads((p/"run_plan.json").read_text())
manifest=json.loads((p/"code_sha256.json").read_text())
mismatches=[name for name,digest in manifest.items() if hashlib.sha256((pathlib.Path(plan["code"])/name).read_bytes()).hexdigest()!=digest]
out={"code_mismatches":mismatches,"runs":[]}
expected={"t1n","t1c","t2w","t2f"}
for d in sorted((p/"runs").iterdir()):
 f=d/"batch_targets.csv"
 if not f.exists(): continue
 with f.open() as stream: rows=list(csv.DictReader(stream))
 groups={}
 for row in rows: groups.setdefault(int(row["step"]),[]).append(row)
 bad=[];complete=[]
 for step,group in groups.items():
  if len(group)!=4:
   if step!=max(groups): bad.append({"step":step,"reason":"incomplete"})
   continue
  if len({r["subject"] for r in group})!=4: bad.append({"step":step,"reason":"duplicate-patient"})
  if {r["target"] for r in group}!=expected: bad.append({"step":step,"reason":"targets"})
  if any({r["condition_1"],r["condition_2"],r["condition_3"]}!=expected-{r["target"]} for r in group): bad.append({"step":step,"reason":"conditioning"})
  complete.append(step)
 commands=json.loads((d/"commands.json").read_text())
 out["runs"].append({"run":d.name,"status":(d/"status.txt").read_text().strip(),"complete_batches":len(complete),"violations":bad[:5],"first_batch":groups[min(complete)],"latest_batch":max(complete),"train_command":commands["train"]})
out["queue"]=subprocess.check_output(["squeue","-u",pathlib.Path.home().name,"-h","-o","%i %T %M %R"],universal_newlines=True)
print(json.dumps(out))
