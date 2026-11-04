#!/bin/bash
# The cup to desk (0, y, 1) along the centre line; at each, the servos' reading and a live frame (frames/c<y>.jpg).
for y in "$@"; do
  curl -s -m 60 -X POST -H 'Content-Type: application/json' -d "{\"x\":0,\"y\":$y,\"z\":${Z:-1}}" http://localhost:8080/api/camera/move >/dev/null
  sleep 2
  m=$(curl -s -m 3 http://localhost:8080/api/state | python3 -c "import json,sys; c=json.load(sys.stdin)['camera']; print(*[round(v,2) for v in c['position']+c['measured']])")
  timeout 4 curl -s -m 3 http://localhost:8080/api/camera/feed -o frames/f.mjpg
  /var/b/shared/work/maxarm/.venv/bin/python -c "
d=open('frames/f.mjpg','rb').read(); e=d.rfind(b'\xff\xd9'); s=d.rfind(b'\xff\xd8',0,e); open('frames/c$y${Z:+-z$Z}.jpg','wb').write(d[s:e+2])"
  echo "$y $m"
done
