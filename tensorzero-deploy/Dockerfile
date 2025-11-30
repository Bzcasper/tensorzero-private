# Dockerfile for TensorZero Gateway (Rust)
FROM rust:bookworm as builder

# Install nightly toolchain (required by tensorzero)
RUN rustup default nightly

WORKDIR /app

# Copy workspace Cargo files
COPY Cargo.toml Cargo.lock ./

# Copy all workspace members
COPY tensorzero-core/ ./tensorzero-core/
COPY gateway/ ./gateway/
COPY internal/ ./internal/
# Copy router-core to the location expected by tensorzero-core/Cargo.toml
COPY tensorzero-deploy/router/ ./tensorzero-deploy/router/

# Build the gateway
RUN cargo build --release --package gateway

# Runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/gateway /usr/local/bin/

EXPOSE 3000

CMD ["gateway", "--config-file", "/app/config/tensorzero.toml"]