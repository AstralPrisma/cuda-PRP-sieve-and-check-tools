# SPDX-License-Identifier: GPL-2.0-or-later
"""Checked GFPDIVSV snapshot reader and single-candidate/expression exporter.

These are computational input files, NOT a Prime Seeker server import format.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path

def read_snapshot(path: Path):
    if path.stat().st_size>512*1024*1024:raise ValueError('Snapshot exceeds resource limit')
    raw=path.read_bytes();body,separator,tail=raw.rpartition(b'sha256 ')
    if not separator or not body.endswith(b'\n') or tail!=hashlib.sha256(body).hexdigest().encode()+b'\n':raise ValueError('Snapshot SHA-256 mismatch')
    rows=body.decode('ascii').splitlines()
    if rows[0]!='GFPDIVSV 1':raise ValueError('Unsupported snapshot version')
    names=['n','family','divisor_min','divisor_max','sign_mask','sieved_to','base_sha256','candidates']
    meta={}
    for key,line in zip(names,rows[1:9]):
        k,value=line.split(' ',1)
        if k!=key:raise ValueError('Invalid metadata order')
        meta[key]=value if key in ('family','base_sha256') else int(value)
    if len(meta)!=8:raise ValueError('Missing metadata')
    n=meta['n'];lo,hi=meta['divisor_min'],meta['divisor_max'];family=meta['family']
    if not 3<=n<=10_000_000 or not 2<=lo<=hi<n or family not in ('!','#') or not 1<=meta['sign_mask']<=3 or not n<=meta['sieved_to']<2**62:raise ValueError('Invalid search bounds')
    allowed=None
    if family=='#':
        composite=bytearray(n+1)
        for p in range(2,math.isqrt(n)+1):
            if not composite[p]:composite[p*p::p]=b'\1'*((n-p*p)//p+1)
        largest=next(p for p in range(n,1,-1) if not composite[p])
        allowed={p for p in range(max(3,lo),min(hi+1,largest)) if not composite[p]}
    candidates=[];previous=(0,0)
    for line in rows[9:]:
        t,sign=map(int,line.split());key=(t,sign);bit=1 if sign==-1 else 2
        if sign not in (-1,1) or not lo<=t<=hi or not meta['sign_mask']&bit or key<=previous or allowed is not None and t not in allowed:raise ValueError('Invalid candidate/domain/order')
        candidates.append((t,sign));previous=key
    if len(candidates)!=meta['candidates']:raise ValueError('Candidate count mismatch')
    meta['snapshot_sha256']=hashlib.sha256(raw).hexdigest()
    return meta,candidates

def expression(meta,t,sign):return f"{meta['n']}{meta['family']}/{t}{'+' if sign==1 else '-'}1"

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input',type=Path);target=ap.add_mutually_exclusive_group(required=True)
    target.add_argument('--out-dir',type=Path);target.add_argument('--expressions',type=Path)
    args=ap.parse_args();meta,rows=read_snapshot(args.input)
    if args.expressions:
        with args.expressions.open('x',encoding='ascii',newline='\n') as output:
            for t,sign in rows:output.write(expression(meta,t,sign)+'\n')
    else:
        args.out_dir.mkdir(parents=True,exist_ok=False)
        marker=args.out_dir/'INCOMPLETE';marker.write_text('Do not use until manifest.json exists.\n',encoding='ascii')
        family='F' if meta['family']=='!' else 'P'
        for t,sign in rows:
            name=f"GFPPS_DIV_{family}_n{meta['n']}_d{t}_{'P' if sign==1 else 'M'}.txt"
            with (args.out_dir/name).open('x',encoding='ascii',newline='\n') as output:output.write(expression(meta,t,sign)+'\n')
        (args.out_dir/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8');marker.unlink()
    print(json.dumps({'candidates':len(rows),'sieved_to':meta['sieved_to'],'kind':meta['family'],'n':meta['n']}))

if __name__=='__main__':main()
