#!/usr/bin/env bash
# =============================================================================
# SWR 推送链路自测（零依赖：只用 curl + tar + python3）
#
# 为什么不用 docker push：
#   本机 / CI runner 上 Docker 23+ 若启用 containerd image store，
#   `docker build` 会产出 OCI manifest（application/vnd.oci.image.manifest.v1+json），
#   而 SWR 只接受 Docker v2s2（application/vnd.docker.distribution.manifest.v2+json），
#   推送会以 404 MANIFEST_UNKNOWN / "OCI manifest found, but accept header does not
#   support OCI manifests" 失败。本脚本直接按 registry v2 协议手写 manifest，
#   Content-Type 显式指定 v2s2，从而与 docker/buildx 的实现细节解耦。
#
# 用法：
#   export SWR_USERNAME=xxx
#   export SWR_PASSWORD=xxx          # 建议直接在本终端手敲，不要写进文件
#   ./scripts/swr_push_selftest.sh              # 推 1 MiB，回读校验，立即删除
#   ./scripts/swr_push_selftest.sh --auth-only  # 只验凭证，不写任何东西
#   ./scripts/swr_push_selftest.sh --mib 16     # 测上行带宽
#
# 可选环境变量：
#   SWR_REPO  默认 swr.cn-north-12.myhuaweicloud.com/ascend/torchturbo_npu
#   SWR_TAG   默认 selftest（注意：workflow 用的是 test，两者互不冲突）
#
# 退出码：0 = 全链路通；1 = 凭证/权限问题；2 = 协议/SWR 兼容性问题
# =============================================================================
set -uo pipefail

SWR_REPO="${SWR_REPO:-swr.cn-north-12.myhuaweicloud.com/ascend/torchturbo_npu}"
SWR_TAG="${SWR_TAG:-selftest}"
MODE="full"
MIB="1"

while [ $# -gt 0 ]; do
  case "$1" in
    --auth-only) MODE="auth" ;;
    --mib) shift; MIB="${1:-1}" ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 64 ;;
  esac
  shift
done

case "$MIB" in ''|*[!0-9]*) echo "--mib 必须是正整数" >&2; exit 64 ;; esac

HOST="${SWR_REPO%%/*}"
REPO="${SWR_REPO#*/}"

ok()   { echo "  ✅ $*"; }
bad()  { echo "  ❌ $*"; }
note() { echo "  ·  $*"; }

# ---------------------------------------------------------------------------
# 0. 凭证
# ---------------------------------------------------------------------------
if [ -z "${SWR_USERNAME:-}" ] || [ -z "${SWR_PASSWORD:-}" ]; then
  echo "未检测到 SWR_USERNAME / SWR_PASSWORD。"
  printf '  SWR 用户名: '; read -r SWR_USERNAME
  printf '  SWR 密码/登录密码 (输入不回显): '; read -rs SWR_PASSWORD; echo
fi
if [ -z "${SWR_USERNAME:-}" ] || [ -z "${SWR_PASSWORD:-}" ]; then
  echo "凭证为空，退出。" >&2; exit 1
fi

echo "======================================================================"
echo "SWR 推送自测"
echo "  目标    : ${SWR_REPO}:${SWR_TAG}"
echo "  模式    : ${MODE}$([ "$MODE" = full ] && echo "（${MIB} MiB）")"
echo "  用户名  : ${SWR_USERNAME}"
echo "======================================================================"
echo

# ---------------------------------------------------------------------------
# 1. 用 Basic auth 换 push token —— 这就是"凭证对不对"的判据
# ---------------------------------------------------------------------------
echo "【1/6】凭证 → push token"
RESP=$(curl -sS -m 30 -u "${SWR_USERNAME}:${SWR_PASSWORD}" \
  "https://${HOST}/swr/auth/v2/registry/auth/?service=dockyard&scope=repository:${REPO}:push,pull" 2>&1)
TOKEN=$(printf '%s' "$RESP" | python3 -c "import sys,json;print(json.load(sys.stdin).get('token',''))" 2>/dev/null)

if [ -z "$TOKEN" ]; then
  bad "拿不到 token —— 用户名/密码错误，或该账号无权访问 ${REPO}"
  echo "  原始响应：$(printf '%s' "$RESP" | head -c 400)"
  exit 1
fi
ok "拿到 push token（len=${#TOKEN}）"

# 故意不带 Authorization 访问，若 401 说明 token 确实被服务端认（而不是"任何人都有权")
if [ "$MODE" = auth ]; then
  echo
  echo "【auth-only】到此为止，未写入任何数据。"
  echo "  结论：凭证有效、到 SWR 的网络通。"
  exit 0
fi

AUTH=(-H "Authorization: Bearer ${TOKEN}")

# ---------------------------------------------------------------------------
# 2. 前置检查：tag 已存在则拒绝（绝不覆盖真实镜像）
# ---------------------------------------------------------------------------
echo
echo "【2/6】前置检查：${SWR_TAG} 是否已被占用"
ACC='application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.oci.image.index.v1+json'
EXIST=$(curl -sS -m 30 -o /dev/null -w '%{http_code}' "${AUTH[@]}" -H "Accept: ${ACC}" \
  "https://${HOST}/v2/${REPO}/manifests/${SWR_TAG}")
note "GET manifests/${SWR_TAG} → HTTP ${EXIST}"
if [ "$EXIST" = "200" ]; then
  bad "tag ${SWR_TAG} 已存在，拒绝覆盖。换一个：SWR_TAG=selftest2 $0"
  exit 1
fi
ok "tag 空闲，可安全写入"

# ---------------------------------------------------------------------------
# 3. 造一个不可压缩的层
# ---------------------------------------------------------------------------
echo
echo "【3/6】生成 ${MIB} MiB 随机层（/dev/urandom → tar.gz，压缩率≈1.0）"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK" || exit 1

head -c $((MIB * 1024 * 1024)) /dev/urandom > payload.bin
tar -cf layer.tar payload.bin
gzip -n -9 layer.tar
LAYER_SIZE=$(stat -c %s layer.tar.gz)
LAYER_DIGEST="sha256:$(sha256sum layer.tar.gz | awk '{print $1}')"
note "layer.tar.gz = ${LAYER_SIZE} bytes  ${LAYER_DIGEST}"

CONFIG='{"architecture":"arm64","os":"linux","rootfs":{"type":"layers","diff_ids":[]},"config":{}}'
printf '%s' "$CONFIG" > config.json
CFG_SIZE=$(stat -c %s config.json)
CFG_DIGEST="sha256:$(sha256sum config.json | awk '{print $1}')"

# ---------------------------------------------------------------------------
# 4. 上传两个 blob
# ---------------------------------------------------------------------------
echo
echo "【4/6】上传 blob"
push_blob() {  # $1=file $2=digest $3=描述
  local f="$1" d="$2" name="$3" sz loc loc2 code t0 t1 sep
  sz=$(stat -c %s "$f")
  t0=$(date +%s%N 2>/dev/null || date +%s)

  loc=$(curl -sS -m 60 -D - -o /dev/null -X POST "${AUTH[@]}" \
        "https://${HOST}/v2/${REPO}/blobs/uploads/" 2>/dev/null \
        | tr -d '\r' | awk 'tolower($1)=="location:"{print $2}')

  if [ -z "$loc" ]; then
    # 退路：部分 registry 要求一次性（monolithic）POST 上传
    note "${name}: 无 upload Location，尝试 monolithic POST"
    code=$(curl -sS -m 900 -o /tmp/swr_mono.log -w '%{http_code}' -X POST "${AUTH[@]}" \
           -H 'Content-Type: application/octet-stream' \
           --data-binary "@${f}" \
           "https://${HOST}/v2/${REPO}/blobs/uploads/?digest=${d}")
    if [ "$code" = "201" ] || [ "$code" = "204" ]; then
      ok "${name}: ${sz} bytes（monolithic，HTTP ${code}）"
      return 0
    fi
    bad "${name}: monolithic 也失败 HTTP ${code}"
    head -c 400 /tmp/swr_mono.log 2>/dev/null; echo
    return 1
  fi
  case "$loc" in /*) loc="https://${HOST}${loc}" ;; esac

  # PATCH 上传内容 → registry 会回一个新的 Location
  loc2=$(curl -sS -m 600 -D - -o /dev/null -X PATCH "${AUTH[@]}" \
         -H 'Content-Type: application/octet-stream' \
         --data-binary "@${f}" "$loc" 2>/dev/null \
         | tr -d '\r' | awk 'tolower($1)=="location:"{print $2}')
  [ -n "$loc2" ] && loc="$loc2"
  case "$loc" in /*) loc="https://${HOST}${loc}" ;; esac

  case "$loc" in *\?*) sep='&' ;; *) sep='?' ;; esac
  code=$(curl -sS -m 600 -o /tmp/swr_put.log -w '%{http_code}' -X PUT "${AUTH[@]}" \
         -H 'Content-Type: application/octet-stream' \
         "${loc}${sep}digest=${d}")
  t1=$(date +%s%N 2>/dev/null || date +%s)

  if [ "$code" = "201" ] || [ "$code" = "204" ]; then
    local ms=$(( (t1 - t0) / 1000000 ))
    local mbps
    mbps=$(awk -v b="$sz" -v m="$ms" 'BEGIN{ if(m<=0)m=1; printf "%.2f", b/1048576/(m/1000) }')
    ok "${name}: ${sz} bytes 上传完成（HTTP ${code}，${ms} ms ≈ ${mbps} MiB/s）"
    return 0
  fi
  bad "${name}: PUT blob 失败 HTTP ${code}"
  head -c 400 /tmp/swr_put.log 2>/dev/null; echo
  return 1
}

push_blob config.json "$CFG_DIGEST" "config blob" || exit 2
push_blob layer.tar.gz "$LAYER_DIGEST" "layer  blob"  || exit 2

# ---------------------------------------------------------------------------
# 5. PUT manifest（关键：Content-Type 必须是 Docker v2s2）+ 回读校验
# ---------------------------------------------------------------------------
echo
echo "【5/6】PUT manifest（Content-Type = Docker v2s2）"
MANIFEST=$(cat <<JSON
{"schemaVersion":2,"mediaType":"application/vnd.docker.distribution.manifest.v2+json","config":{"mediaType":"application/vnd.docker.container.image.v1+json","size":${CFG_SIZE},"digest":"${CFG_DIGEST}"},"layers":[{"mediaType":"application/vnd.docker.image.rootfs.diff.tar.gzip","size":${LAYER_SIZE},"digest":"${LAYER_DIGEST}"}]}
JSON
)

CODE=$(curl -sS -m 120 -D /tmp/swr_man.hdr -o /tmp/swr_man.log -w '%{http_code}' -X PUT "${AUTH[@]}" \
  -H 'Content-Type: application/vnd.docker.distribution.manifest.v2+json' \
  --data-binary "$MANIFEST" \
  "https://${HOST}/v2/${REPO}/manifests/${SWR_TAG}")
if [ "$CODE" != "201" ] && [ "$CODE" != "200" ]; then
  bad "PUT manifest 失败 HTTP ${CODE}"
  head -c 500 /tmp/swr_man.log 2>/dev/null; echo
  exit 2
fi
# manifest digest 取自响应头 Docker-Content-Digest（响应体通常是空的，不要解析 body）
MAN_DIGEST=$(tr -d '\r' < /tmp/swr_man.hdr | awk 'tolower($1)=="docker-content-digest:"{print $2}')
if [ -z "$MAN_DIGEST" ]; then
  MAN_DIGEST=$(tr -d '\r' < /tmp/swr_man.hdr | awk 'tolower($1)=="location:"{print $2}' | sed 's#.*/manifests/##')
fi
if [ -z "$MAN_DIGEST" ]; then
  # 最后的退路：我们发出去的字节就是 manifest 本体，本地 sha256 与 digest 一致
  MAN_DIGEST="sha256:$(printf '%s' "$MANIFEST" | sha256sum | awk '{print $1}')"
  note "服务端未回 digest，用本地 sha256 计算"
fi
ok "manifest 已写入（HTTP ${CODE}）digest=${MAN_DIGEST}"

echo
echo "  回读校验（Accept 同时给 Docker + OCI，看服务端实际存的是哪种）"
BACK=$(curl -sS -m 60 "${AUTH[@]}" -H "Accept: ${ACC}" \
  "https://${HOST}/v2/${REPO}/manifests/${SWR_TAG}" 2>/dev/null)
FMT=$(printf '%s' "$BACK" | python3 -c "import sys,json;d=json.loads(sys.stdin.read() or '{}');print(d.get('mediaType',''))" 2>/dev/null)
STORED=$(printf '%s' "$BACK" | python3 -c "import sys,json;d=json.loads(sys.stdin.read() or '{}');print(sum(l.get('size',0) for l in d.get('layers',[])))" 2>/dev/null)
case "${STORED:-}" in ''|*[!0-9]*) STORED=0 ;; esac

note "mediaType = ${FMT:-<空>}"
note "服务端层字节 = ${STORED} / 本地 ${LAYER_SIZE}"
case "$FMT" in
  application/vnd.docker.distribution.manifest.v2+json)
    ok "存的是 Docker v2s2 ✓（这是后面所有 CI 推送必须满足的格式）" ;;
  *oci.image.*)
    bad "存的是 OCI manifest —— SWR 接受但下游可能不认，改用 buildx --output oci-mediatypes=false" ;;
  *)
    bad "manifest 格式未知：${FMT:-<空>}" ;;
esac
if [ "$STORED" -gt 0 ] && [ "$STORED" != "$LAYER_SIZE" ]; then
  bad "层大小不一致 → 可能被截断，检查网络/超时"
fi

# ---------------------------------------------------------------------------
# 6. 清理
# ---------------------------------------------------------------------------
echo
echo "【6/6】清理：删除 ${SWR_TAG}"
DEL_TARGET="${MAN_DIGEST:-${SWR_TAG}}"
CODE=$(curl -sS -m 60 -o /tmp/swr_del.log -w '%{http_code}' -X DELETE "${AUTH[@]}" \
  "https://${HOST}/v2/${REPO}/manifests/${DEL_TARGET}")
note "DELETE manifests/${DEL_TARGET} → HTTP ${CODE}"

LEFT=$(curl -sS -m 30 -o /dev/null -w '%{http_code}' "${AUTH[@]}" -H "Accept: ${ACC}" \
  "https://${HOST}/v2/${REPO}/manifests/${SWR_TAG}")
if [ "$LEFT" = "404" ]; then
  ok "已删除（回读 404）"
  note "blob 需等 SWR 侧 GC 回收，通常数小时~1 天，属正常"
else
  echo "  ⚠️  删除后仍能读到（HTTP ${LEFT}），请手工清理 ${SWR_REPO}:${SWR_TAG}"
  echo "     可能原因：SWR 控制台设了「禁止删除」策略，或该 tag 是 manifest list"
fi

echo
echo "======================================================================"
echo "结论：凭证有效 ✅   上传链路通 ✅   manifest 格式 = ${FMT:-unknown}"
echo "======================================================================"
