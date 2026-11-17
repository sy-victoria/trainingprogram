FROM python:3.11.11-slim-bookworm@sha256:081075da77b2b55c23c088251026fb69a7b2bf92471e491ff5fd75c192fd38e5
COPY --from=docker:29.8.2-cli@sha256:35cfcf893c20f2931f4909151b5407aaaeab711277de7ce8b8f521f095d54c32 /usr/local/bin/docker /usr/local/bin/docker
ARG SWEBENCH_REV=02e7a74ffd0b707aab73d203fe87bdc7c76afc8e
RUN apt-get update && apt-get install -y --no-install-recommends git patch ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Build context must contain an archive produced with git archive at SWEBENCH_REV.
ADD swebench-source.tar /opt/swebench/
COPY swebench-worker-resolved.txt /opt/requirements-lock.txt
RUN pip install --no-cache-dir -r /opt/requirements-lock.txt \
    && pip check \
    && pip freeze > /opt/requirements-resolved.txt \
    && printf '%s\n' "$SWEBENCH_REV" > /opt/swebench-revision.txt
ENV PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface
WORKDIR /results
# Trusted harness only: never mount Docker socket into the untrusted task container.
ENTRYPOINT ["python", "-m", "swebench.harness.run_evaluation"]
