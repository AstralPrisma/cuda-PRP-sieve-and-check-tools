import argparse,hashlib,json,math,pathlib,re,subprocess,tempfile,time,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"scripts"))
from ghcw_to_cands import read_snapshot,read
ROOT=pathlib.Path(__file__).resolve().parents[1]
def prime(n):
 if n<2:return False
 for p in [2,3,5,7,11,13,17,19,23,29,31,37]:
  if n%p==0:return n==p
 d=n-1;s=0
 while d%2==0:s+=1;d//=2
 for a in [2,325,9375,28178,450775,9780504,1795265022]:
  if a%n==0:continue
  x=pow(a,d,n)
  if x in [1,n-1]:continue
  for _ in range(s-1):
   x=x*x%n
   if x==n-1:break
  else:return False
 return True
def eligible(b,n,c):
 if b%2 and n%2:return False
 d=math.gcd(b,n)
 if c==-1 and d>1:return False
 while d%2==0:d//=2
 return not(c==1 and d>1)
def expected(b,rows,low,high):
 ps=[p for p in range(low+1,high+1) if prime(p)];out=set()
 for n,c in rows:
  for p in ps:
   if (pow(b,n,p)*pow(n,b,p)+c)%p:continue
   if n*math.log2(b)+b*math.log2(n)<64 and b**n*n**b+c==p:continue
   break
  else:out.add((n,c))
 return out
def seed(path,b,rows,p,mode=0):
 rows=sorted(rows)
 header=f'ABC {b}^$a*$a^{b}'+('$b // GHCWSV v2' if mode==0 else f'{mode:+d} // GHCWSV v1')+f' sieved_to={p} count={len(rows)}\n'
 body=header+''.join(f'{n} {c:+d}\n' if mode==0 else f'{n}\n' for n,c in rows)
 path.write_bytes((body+'#SHA256 '+hashlib.sha256(body.encode()).hexdigest()+'\n').encode())
class Runner:
 def __init__(self,system,arch):self.exe=pathlib.Path(system).resolve();self.count=0
 def run(self,args,ok=0):
  cmd=[str(self.exe)]
  cmd += list(map(str,args))
  r=subprocess.run(cmd,capture_output=True,timeout=45);self.count+=1
  if r.returncode!=ok:raise AssertionError((cmd,r.returncode,r.stdout.decode('utf8','replace')[-1500:],r.stderr.decode('utf8','replace')))
  r.stdout.decode('utf8');return r.stdout.decode('utf8')
def tests(system,arch,gpu):
 runner=Runner(system,arch)
 with tempfile.TemporaryDirectory(prefix='both-test-') as folder:
  folder=pathlib.Path(folder)
  def path(n):return folder/n
  for b in [2,3,7,16,325,65537]:
   want=expected(b,[(n,c) for n in range(2,602) for c in [-1,1] if eligible(b,n,c)],1,997)
   modes=['auto','direct','transform'] if gpu else ['auto']
   for algorithm in modes:
    target=path(f'{b}-{algorithm}.txt');args=['-b',b,'-n',2,'-N',601,'--sign','both','-P',997,'-o',target,'--algorithm',algorithm,'--prime-threads',1,'--prime-generator','segmented','--batch-primes',31]
    if not gpu:args+=['--cpu-reference']
    text=runner.run(args);snapshot=read_snapshot(target);assert set(snapshot.candidates())==want,(b,algorithm)
    assert snapshot.sign==0 and snapshot.p==997 and snapshot.count==len(want)
    assert 'plus=' in text and 'minus=' in text
   union=set()
   for c in [-1,1]:
    target=path(f'{b}-{c}.txt');runner.run(['-b',b,'-n',2,'-N',601,'--sign',str(c),'-P',997,'-o',target,'--cpu-reference'])
    legacy=read(target);assert legacy[1]==c
    union.update(read_snapshot(target).candidates())
   assert union==want
  # Sign pruning must keep both small primes 71 and 73 for b=2,n=3.
  assert {(3,-1),(3,1)}<=set(read_snapshot(path('2-auto.txt')).candidates())
  # Resume a v1 file without changing its mode or allowing --sign both to add work.
  legacy=path('legacy.txt');seed(legacy,7,[(n,1) for n in range(2,42,2)],97,1)
  out=path('legacy-out.txt');runner.run(['-i',legacy,'-P',997,'-o',out,'--cpu-reference']);assert read_snapshot(out).sign==1
  before=legacy.read_bytes();runner.run(['-i',legacy,'--sign','both','-P',997,'-o',legacy,'--cpu-reference'],1);assert legacy.read_bytes()==before
  # Mixed masks: one sign already removed, same-n ordering and empty states.
  rows=[(2,1),(3,-1),(3,1),(5,-1),(6,1),(8,1),(11,-1)]
  initial=path('mixed.txt');seed(initial,2,rows,97);final=path('mixed-out.txt')
  runner.run(['-i',initial,'-P',997,'-o',final,'--cpu-reference']);assert set(read_snapshot(final).candidates())==expected(2,rows,97,997)
  empty=path('empty.txt');seed(empty,7,[],997);runner.run(['-i',empty,'-P',2000,'-o',empty,'--sign','both','--cpu-reference']);assert read_snapshot(empty).count==0
  runner.run(['-i',initial,'--sign','+1','-P',997,'-o',path('bad-sign.txt'),'--cpu-reference'],1)
  for index,bad in enumerate([initial.read_bytes()[:-5],initial.read_bytes().replace(b'3 -1',b'4 -1'),initial.read_bytes()+b'extra\n']):
   malformed=path(f'corrupt{index}.txt');malformed.write_bytes(bad)
   runner.run(['-i',malformed,'-P',997,'-o',path('no-write.txt'),'--cpu-reference'],1)
   try:read_snapshot(malformed)
   except ValueError:pass
   else:raise AssertionError('converter accepted corrupt state')
  for index,rows in enumerate([[(3,1),(3,1)],[(3,1),(3,-1)],[(3,0)],[(1,1)]]):
   body='ABC 2^$a*$a^2$b // GHCWSV v2 sieved_to=97 count='+str(len(rows))+'\n'+''.join(f'{n} {c:+d}\n' for n,c in rows)
   invalid=path(f'badrows{index}.txt');invalid.write_bytes((body+'#SHA256 '+hashlib.sha256(body.encode()).hexdigest()+'\n').encode())
   runner.run(['-i',invalid,'-P',997,'-o',path('no-write.txt'),'--cpu-reference'],1)
   try:read_snapshot(invalid)
   except ValueError:pass
   else:raise AssertionError('converter accepted invalid ordering/sign')
  # Coefficient and inverse-transform paths versus an independent modular oracle.
  cases=[(2,range(2,62),5000,7000),(65537,range(2,30),10**12,10**12+1000),(2,range(2**32-9,2**32),2**62-2000,2**62-1),(4,[4],65520,65540)]
  for index,(b,ns,low,high) in enumerate(cases):
   rows=[(n,c) for n in ns for c in [-1,1] if eligible(b,n,c)];seeded=path(f'seed{index}.txt');seed(seeded,b,rows,low);want=expected(b,rows,low,high)
   for algorithm in (['auto','direct','transform'] if gpu else ['auto']):
    out=path(f'high{index}{algorithm}.txt');args=['-i',seeded,'-o',out,'-P',high,'--algorithm',algorithm,'--prime-generator','mr','--prime-threads',1,'--batch-primes',17]
    if not gpu:args+=['--cpu-reference']
    runner.run(args);assert set(read_snapshot(out).candidates())==want,(index,algorithm)
  if gpu:
   factor_path=path('factors.txt');target=path('with-factors.txt')
   runner.run(['-b',2,'-n',2,'-N',101,'--sign','both','-P',997,'-o',target,'-O',factor_path,'--prime-threads',1,'--prime-generator','segmented'])
   signs=set()
   for line in factor_path.read_text().splitlines():
    m=re.fullmatch(r'(\d+) \| 2\^(\d+)\*(\d+)\^2([+-])1',line);assert m and m[2]==m[3]
    p,n,c=int(m[1]),int(m[2]),1 if m[4]=='+' else -1;assert prime(p) and (pow(2,n,p)*pow(n,2,p)+c)%p==0;signs.add(c)
   assert signs=={-1,1}
  # Network conversion preserves both signs and filters before writing files.
  for mode in ['both','+1','-1']:
   dest=path('cands'+mode);subprocess.run([sys.executable,str(ROOT/'scripts/ghcw_to_cands.py'),str(path('2-auto.txt')),'--out-dir',str(dest),'--network','--nmin','2','--nmax','15','--sign='+mode],capture_output=True,check=True)
   got={f.read_text().strip().splitlines()[-1] for f in dest.iterdir()};want={f'2^{n}*{n}^2{c:+d}' for n,c in read_snapshot(path('2-auto.txt')).candidates() if n<=15 and (mode=='both' or c==int(mode))};assert got==want
 return {'platform':system,'arch':arch,'gpu':gpu,'commands':runner.count,'result':'PASS'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description='Bounded GHCWSV single/both-sign regression; GPU only with --gpu')
 p.add_argument('--binary',type=pathlib.Path,required=True);p.add_argument('--gpu',action='store_true');a=p.parse_args()
 report=tests(str(a.binary),0,a.gpu);print(json.dumps(report),flush=True)
