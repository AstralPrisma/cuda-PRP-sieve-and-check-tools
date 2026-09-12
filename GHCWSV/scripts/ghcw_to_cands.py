#!/usr/bin/env python3
"""Validate GHCWSV v1/v2 and emit one expression per file. No network writes."""
import argparse,array,hashlib,pathlib,re
from dataclasses import dataclass
PATTERN=re.compile(r'ABC ([0-9]+)\^\$a\*\$a\^([0-9]+)(\+1|-1|\$b) // GHCWSV v([12]) sieved_to=([0-9]+) count=([0-9]+)')
MAX_BYTES=320*1024**2

@dataclass
class Snapshot:
    b:int
    sign:int
    p:int
    ns:array.array
    masks:array.array
    count:int
    def candidates(self):
        for n,mask in zip(self.ns,self.masks):
            if mask&2:yield n,-1
            if mask&1:yield n,1

def read_snapshot(path):
    path=pathlib.Path(path)
    if path.stat().st_size>MAX_BYTES:raise ValueError('input exceeds320MiB')
    digest=hashlib.sha256();bytes_read=0
    with path.open(encoding='utf8',newline='') as stream:
        def line():
            nonlocal bytes_read
            raw=stream.readline(512);bytes_read+=len(raw.encode('utf8'))
            if bytes_read>MAX_BYTES or (raw and not raw.endswith('\n')):raise ValueError('oversized/truncated candidate line')
            return raw.replace('\r','')
        header=line();m=PATTERN.fullmatch(header.rstrip('\n'))
        if not m:raise ValueError('unsupported header')
        both=m[3]=='$b'
        if (both and m[4]!='2') or (not both and m[4]!='1'):raise ValueError('sign/version mismatch')
        b,other=int(m[1]),int(m[2]);sign=0 if both else 1 if m[3]=='+1' else -1
        p,count=int(m[5]),int(m[6])
        if b!=other or not 2<=b<=2**32-1 or not 1<=p<=2**62-1:raise ValueError('invalid header values')
        if count>10000000*(2 if both else 1):raise ValueError('too many candidates')
        result=Snapshot(b,sign,p,array.array('I'),array.array('B'),count)
        digest.update(header.encode());previous=(0,0);rows=0
        while True:
            row=line()
            if row.startswith('#SHA256 '):
                if row!='#SHA256 '+digest.hexdigest()+'\n' or stream.read(1):raise ValueError('checksum/footer mismatch')
                break
            if not row:raise ValueError('missing checksum footer')
            digest.update(row.encode());text=row[:-1]
            if both:
                match=re.fullmatch(r'([0-9]+) ([+-]1)',text)
                if not match:raise ValueError('both-sign rows require n and +1/-1')
                n,c=int(match[1]),int(match[2])
            else:
                if not re.fullmatch(r'[0-9]+',text):raise ValueError('invalid candidate row')
                n,c=int(text),sign
            if not 2<=n<=2**32-1 or (n,c)<=previous:raise ValueError('candidate range/order mismatch')
            bit=1 if c==1 else 2
            if result.ns and n==result.ns[-1]:result.masks[-1]|=bit
            else:result.ns.append(n);result.masks.append(bit)
            rows+=1;previous=(n,c)
            if rows>count or len(result.ns)>10000000:raise ValueError('extra/oversized candidate data')
        if rows!=count:raise ValueError('truncated candidate list')
        return result
def read(path):
    """Backward-compatible single-sign API; use read_snapshot for both signs."""
    result=read_snapshot(path)
    if result.sign==0:raise ValueError('both-sign snapshot: use read_snapshot().candidates()')
    return result.b,result.sign,list(result.ns)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=pathlib.Path);p.add_argument('--out-dir',type=pathlib.Path,required=True)
    p.add_argument('--network',action='store_true',help='emit Prime Seeker TYPE + expression files');p.add_argument('--nmin',type=int,default=2);p.add_argument('--nmax',type=int,default=2**32-1)
    p.add_argument('--sign',choices=['both','+1','-1'],default='both',help='filter exported signs (default: all signs present in the file)');a=p.parse_args()
    if a.nmin>a.nmax:p.error('nmin exceeds nmax')
    snapshot=read_snapshot(a.input);b=snapshot.b
    if a.out_dir.exists():p.error('output directory must not already exist')
    a.out_dir.mkdir(parents=True)
    count=0
    for n,c in snapshot.candidates():
        if not a.nmin<=n<=a.nmax or (a.sign!='both' and c!=int(a.sign)):continue
        path=a.out_dir/f'{"GHCWPS_" if a.network else ""}cand_{b}_{n}_{"S" if c==1 else "R"}.txt'
        with path.open('x',encoding='utf8',newline='\n') as f:f.write(('GHCWPS\n' if a.network else '')+f'{b}^{n}*{n}^{b}{c:+d}\n')
        count+=1
    print(f'Created {count} candidate files; no tasks uploaded.')
if __name__=='__main__':main()
