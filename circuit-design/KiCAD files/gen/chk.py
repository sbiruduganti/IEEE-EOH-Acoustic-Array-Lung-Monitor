import re,json,sys
t=open(sys.argv[1]).read()
want=json.load(open(sys.argv[2]))
got={}
for m in re.finditer(r'\(net \(code "?\d+"?\) \(name "([^"]*)"\)(.*?)\)\s*(?=\(net |\)\s*\)\s*$)',t,re.S):
    name=m.group(1).split('/')[-1]
    for n in re.finditer(r'\(node \(ref "([^"]+)"\) \(pin "([^"]+)"\)',m.group(2)):
        got["%s.%s"%(n.group(1),n.group(2))]=name
bad=0
for k,v in want.items():
    g=got.get(k)
    if g!=v and not (g and g.startswith("unconnected")):
        bad+=1
        if bad<30: print("MISMATCH",k,"want",v,"got",g)
    elif g is None or g.startswith("unconnected"):
        bad+=1; print("MISSING",k,v,g)
# nets with single node
nets={}
for k,v in got.items(): nets.setdefault(v,[]).append(k)
single=[(n,p) for n,p in nets.items() if len(p)==1 and not n.startswith("unconnected")]
print("checked",len(want),"bad",bad,"nets",len(nets),"single-node nets:",single)
