# SPDX-License-Identifier: GPL-2.0-or-later
"""Convert a checked GFPDIVSV snapshot to canonical Prime Seeker GFPPS tasks."""
import argparse,hashlib,json,os
from pathlib import Path
from candidates import read_snapshot,expression

def server_tasks(meta,rows):
    # read_snapshot has already validated n, divisors, prime domain and signs.
    for divisor,sign in rows:
        candidate=expression(meta,divisor,sign)
        name=f"GFPPS_cand_DIV_{meta['n']}_{divisor}_{'F'if meta['family']=='!'else'P'}_{'S'if sign==1 else'R'}.txt"
        yield {'filename':name,'content':f'GFPPS\n{candidate}\n'}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('input',type=Path)
    dest=ap.add_mutually_exclusive_group(required=True);dest.add_argument('--out-dir',type=Path);dest.add_argument('--manifest',type=Path)
    args=ap.parse_args();meta,rows=read_snapshot(args.input)
    if args.out_dir:
        args.out_dir.mkdir(parents=True,exist_ok=False);marker=args.out_dir/'INCOMPLETE';marker.write_text('Do not import until manifest.json is present.\n')
        for task in server_tasks(meta,rows):
            with(args.out_dir/task['filename']).open('x',encoding='ascii',newline='\n')as f:f.write(task['content'])
        (args.out_dir/'manifest.json').write_text(json.dumps({'format':'GFPPS-factor-removal-v1','source':meta,'files':len(rows)},indent=2)+'\n');marker.unlink()
    else:
        with args.manifest.open('x',encoding='ascii',newline='\n')as f:
            for task in server_tasks(meta,rows):f.write(json.dumps(task,separators=(',',':'))+'\n')
            f.flush();os.fsync(f.fileno())
    print(json.dumps({'tasks':len(rows),'sieved_to':meta['sieved_to'],'snapshot_sha256':meta['snapshot_sha256'],'uploaded':False,'credit_awarded':False}))
if __name__=='__main__':main()
