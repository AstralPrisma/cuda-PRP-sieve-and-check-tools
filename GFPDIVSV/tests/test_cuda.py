# SPDX-License-Identifier: GPL-2.0-or-later
"""Bounded CUDA correctness tests; no accounts, uploads or production data."""
from pathlib import Path
import argparse,hashlib,json,math,os,signal,subprocess,sys,tempfile,time

tool=Path(__file__).resolve().parents[1];sys.path.insert(0,str(tool/'scripts'))
from candidates import read_snapshot
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--binary',type=Path,required=True);ap.add_argument('--report',type=Path)
args=ap.parse_args();binary=args.binary.resolve();checks=[]
def primes(n):
    mark=bytearray(n+1)
    for p in range(2,math.isqrt(n)+1):
        if not mark[p]:mark[p*p::p]=b'\1'*((n-p*p)//p+1)
    return [i for i in range(2,n+1)if not mark[i]]
small=primes(10000)
def reference(n,family,signs=(-1,1)):
    factors=list(range(2,n+1))if family=='factorial'else[p for p in small if p<=n]
    a=math.prod(factors);domain=range(2,n)if family=='factorial'else[p for p in factors if 2<p<factors[-1]]
    return {(d,s)for d in domain for s in signs if all(a//d+s==p or(a//d+s)%p for p in small)}
with tempfile.TemporaryDirectory(prefix='gfpdivsv-cuda-check-')as directory:
    work=Path(directory)
    def run(*argv,ok=True):
        result=subprocess.run([str(binary),*map(str,argv)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        if ok:assert result.returncode==0,result.stdout.decode(errors='replace')[-2500:]
        else:assert result.returncode!=0
        return result.stdout.decode(errors='replace')
    assert 'CUDA_SELFTEST_PASS comparisons=2816'in run('--selftest')
    for n in range(3,46):
        for family in('factorial','primorial'):
            path=work/f'{family}-{n}.txt'
            run('--n',n,'--family',family,'--sign','both','-P',10000,'-o',path,'--verify-cuda')
            meta,rows=read_snapshot(path);assert meta['sieved_to']==10000 and set(rows)==reference(n,family),(n,family)
            checks.append([n,family,len(rows)])
    for sign in('+','-'):
        path=work/f'single-{sign}.txt';run('--n',31,'--family','factorial','--sign',sign,'-P',10000,'-o',path,'--verify-cuda')
        assert set(read_snapshot(path)[1])==reference(31,'factorial',(1 if sign=='+'else-1,))
    for chains in(1,2,4,8):
        for packing in('raw','valuation'):
            path=work/f'chain-{chains}-{packing}.txt'
            run('--n',97,'--family','factorial','-P',10000,'-o',path,'--gpu-chains',chains,'--gpu-product',packing,'--batch-primes',131,'--gpu-tile-chunks',8,'--verify-cuda')
            assert set(read_snapshot(path)[1])==reference(97,'factorial')
    for n,family in[(100,'factorial'),(101,'primorial')]:
        path=work/f'resume-{n}.txt';direct=work/f'direct-{n}.txt'
        run('--n',n,'--family',family,'-P',1000,'-o',path)
        run('-i',path,'-P',10000,'--verify-cuda')
        run('--n',n,'--family',family,'-P',10000,'-o',direct)
        assert path.read_bytes()==direct.read_bytes()
        damaged=work/f'bad-{n}.txt';damaged.write_bytes(path.read_bytes().replace(b'candidates ',b'Candidates '))
        run('-i',damaged,'-P',10001,ok=False)
        run('-i',path,'--n',n+1,'-P',10001,ok=False)
        before=path.read_bytes();run('--n',n,'--family',family,'-P',10000,'-o',path,ok=False);assert path.read_bytes()==before
    text=run('--n',100,'--family','factorial','-P',100000,'-o',work/'partial.txt','--max-batches',1,'--batch-primes',131)
    assert 'result=PARTIAL'in text and read_snapshot(work/'partial.txt')[0]['sieved_to']<100000
    # An obsolete host-only mode is not accepted by this CUDA-only release.
    run('--n',100,'--family','factorial','--backend','cpu',ok=False)
    state=work/'stop.txt';stop=work/'stop.request';log=work/'stop.log'
    with log.open('wb')as stream:
        p=subprocess.Popen([str(binary),'--n','10000','--family','factorial','-P','1e10','-o',str(state),'--stop-file',str(stop),'--checkpoint-seconds','1'],stdout=stream,stderr=stream)
        try:
            deadline=time.monotonic()+25
            while b'progress:'not in log.read_bytes():
                assert p.poll()is None and time.monotonic()<deadline
                time.sleep(.03)
            if os.name=='nt':stop.touch()
            else:p.send_signal(signal.SIGINT)
            assert p.wait(timeout=25)==130
        finally:
            if p.poll()is None:
                stop.touch();p.wait(timeout=25)
    meta,_=read_snapshot(state);assert 10000<meta['sieved_to']<10000000000
    end=meta['sieved_to']+1000
    run('-i',state,'-P',end)
    direct=work/'stop-direct.txt';run('--n',10000,'--family','factorial','-P',end,'-o',direct)
    assert state.read_bytes()==direct.read_bytes()
    source=work/'factorial-31.txt';out=work/'tasks'
    subprocess.run([sys.executable,str(tool/'scripts/gfpdivsv_to_tasks.py'),str(source),'--out-dir',str(out)],check=True,capture_output=True)
    assert len(list(out.glob('*.txt')))==len(read_snapshot(source)[1])
report={'ok':True,'program':'GFPDIVSV1.1 CUDA','binary':binary.name,'sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
        'exhaustive_searches':86,'selftest_modular_comparisons':2816,
        'checks':['independent exact integer oracle','single and dual signs','raw and valuation products','chain1/2/4/8','unaligned batch tails','checked continuation','damage and mismatch rejection','safe interruption and resume','partial frontier','offline conversion','no host-only sieve mode']}
if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
