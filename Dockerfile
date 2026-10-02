# DensForge: CPU-only numerical computing. No CUDA layer -- the method is
# O(N*k*d) per fit plus one O(N^3) eigendecomposition, so a GPU would add
# gigabytes of dependencies without adding capability.
FROM python:3.13-slim

# Single-threaded BLAS. Not a performance choice: multi-threaded reductions sum
# in a nondeterministic order, which perturbs the last bits and makes the
# bit-exactness invariants flaky rather than informative.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependencies first, so a source-only change does not reinstall the scientific
# stack. --no-cache-dir keeps the wheels out of the layer.
COPY requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY pyproject.toml README.md LICENSE CHANGELOG.md ./
COPY densforge ./densforge
COPY examples ./examples
COPY tests ./tests
COPY docs ./docs
COPY Makefile ./

# Run as a non-root user. The `useradd` is in the same layer as the WORKDIR
# ownership fix so no layer needs root at build time after this one.
RUN useradd --create-home --uid 10001 densforge \
    && chown -R densforge:densforge /app
USER densforge

# Fail the build if the package cannot even be imported, rather than discovering
# it when a container is already deployed.
RUN python -c "import densforge; print('densforge', densforge.__version__)"

# A fast, honest smoke check. `--quick` shrinks the sample counts but keeps 3
# seeds, because a container that cannot run a real measurement should not look
# like one that can.
HEALTHCHECK --interval=60s --timeout=120s --start-period=10s --retries=2 \
    CMD python -c "import densforge; assert densforge.DensFuse.available()" || exit 1

ENTRYPOINT ["python", "examples/run_demo.py"]
CMD ["--quick"]
