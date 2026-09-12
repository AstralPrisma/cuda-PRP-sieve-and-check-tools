#!/usr/bin/env python3
"""Validate GHCWSV v1 and emit one expression per file. No network writes."""
import argparse,hashlib,pathlib,re
PATTERN=re.compile(r'ABC (\d+)\^\$a\*\$a\^(\d+)([+-])1 // GHCWSV v1 sieved_to=(\d+) count=(\d+)')
def read(path):
    if path.stat().st_size>128*1024**2:raise ValueError('input exceeds128MiB')
    text=path.read_text(encoding='utf8').replace('\r','')
    body,footer=text.rsplit('#SHA256 ',1)
    if footer!=hashlib.sha256(body.encode()).hexdigest()+'\n':raise ValueError('checksum/footer mismatch')
    lines=body.splitlines();m=PATTERN.fullmatch(lines[0])
    if not m:raise ValueError('unsupported header')
    b,other=int(m[1]),int(m[2]);sign=1 if m[3]=='+' else -1
    if b!=other or not 2<=b<=2**32-1 or not 1<=int(m[4])<=2**62-1:raise ValueError('invalid header values')
    if any(not re.fullmatch('[0-9]+',s) for s in lines[1:]):raise ValueError('invalid candidate row')
    ns=list(map(int,lines[1:]))
    if len(ns)!=int(m[5]) or len(ns)>10000000 or any(not 2<=n<=2**32-1 for n in ns):raise ValueError('count/range mismatch')
    if any(a>=z for a,z in zip(ns,ns[1:])):raise ValueError('candidates not strictly increasing')
    return b,sign,ns
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=pathlib.Path);p.add_argument('--out-dir',type=pathlib.Path,required=True)
    p.add_argument('--network',action='store_true',help='emit Prime Seeker TYPE + expression files');p.add_argument('--nmin',type=int,default=2);p.add_argument('--nmax',type=int,default=2**32-1);a=p.parse_args()
    if a.nmin>a.nmax:p.error('nmin exceeds nmax')
    b,c,ns=read(a.input);ns=[n for n in ns if a.nmin<=n<=a.nmax]
    if a.out_dir.exists():p.error('output directory must not already exist')
    a.out_dir.mkdir(parents=True)
    for n in ns:
        path=a.out_dir/f'{"GHCWPS_" if a.network else ""}cand_{b}_{n}_{"S" if c==1 else "R"}.txt'
        with path.open('x',encoding='utf8',newline='\n') as f:f.write(('GHCWPS\n' if a.network else '')+f'{b}^{n}*{n}^{b}{c:+d}\n')
    print(f'Created {len(ns)} candidate files; no tasks uploaded.')
if __name__=='__main__':main()
