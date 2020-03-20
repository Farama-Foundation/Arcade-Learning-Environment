FROM alpine:latest

RUN apk add --no-cache \
  bash \
  ca-certificates \
  curl \
  wget \
  jq

COPY download-asset.sh /usr/bin/download-asset
RUN chmod +x /usr/bin/download-asset

ENTRYPOINT ["/usr/bin/download-asset"]
