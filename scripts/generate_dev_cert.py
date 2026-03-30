import argparse
import ipaddress
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


DEFAULT_CERTS_DIR = Path(__file__).resolve().parents[1] / "certs"


def get_local_ipv4_addresses() -> list[str]:
    discovered = {"127.0.0.1"}
    hostname = socket.gethostname()
    try:
        for result in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            discovered.add(result[4][0])
    except socket.gaierror:
        pass
    return sorted(discovered)


def write_pem(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def build_name(common_name: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def create_ca(ca_cert_path: Path, ca_key_path: Path, force: bool) -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
    if ca_cert_path.exists() and ca_key_path.exists() and not force:
        return (
            x509.load_pem_x509_certificate(ca_cert_path.read_bytes()),
            serialization.load_pem_private_key(ca_key_path.read_bytes(), password=None),
        )

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(timezone.utc)
    subject = issuer = build_name("Crowd Management Dev CA")
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()), critical=False)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )

    write_pem(
        ca_key_path,
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ),
    )
    write_pem(ca_cert_path, cert.public_bytes(serialization.Encoding.PEM))
    return cert, key


def create_server_cert(
    ca_cert: x509.Certificate,
    ca_key: rsa.RSAPrivateKey,
    cert_path: Path,
    key_path: Path,
    hostnames: list[str],
    ip_addresses: list[str],
) -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(timezone.utc)

    san_entries = [x509.DNSName(host) for host in hostnames]
    san_entries.extend(x509.IPAddress(ipaddress.ip_address(ip)) for ip in ip_addresses)

    cert = (
        x509.CertificateBuilder()
        .subject_name(build_name(hostnames[0]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=825))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )

    write_pem(
        key_path,
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ),
    )
    write_pem(cert_path, cert.public_bytes(serialization.Encoding.PEM))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate local HTTPS certificates for the Crowd Management System.")
    parser.add_argument("--certs-dir", default=str(DEFAULT_CERTS_DIR), help="Directory to write generated PEM files.")
    parser.add_argument("--hostname", action="append", default=["localhost"], help="Hostname to include in the server certificate.")
    parser.add_argument("--ip", action="append", default=[], help="Extra IP address to include in the server certificate.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing certificates.")
    args = parser.parse_args()

    certs_dir = Path(args.certs_dir)
    ca_cert_path = certs_dir / "dev-ca.pem"
    ca_key_path = certs_dir / "dev-ca-key.pem"
    cert_path = certs_dir / "dev-cert.pem"
    key_path = certs_dir / "dev-key.pem"

    local_ips = get_local_ipv4_addresses()
    all_ips = sorted({*local_ips, *args.ip})
    hostnames = list(dict.fromkeys(args.hostname))

    ca_cert, ca_key = create_ca(ca_cert_path, ca_key_path, force=args.force)
    if args.force or not cert_path.exists() or not key_path.exists():
        create_server_cert(ca_cert, ca_key, cert_path, key_path, hostnames, all_ips)

    print("HTTPS development certificates are ready.")
    print(f"CA certificate: {ca_cert_path}")
    print(f"CA private key: {ca_key_path}")
    print(f"Server certificate: {cert_path}")
    print(f"Server private key: {key_path}")
    print("")
    print("Included hostnames:")
    for host in hostnames:
        print(f"  - {host}")
    print("Included IP addresses:")
    for ip in all_ips:
        print(f"  - {ip}")
    print("")
    print("Trust the CA certificate on any phone that will open the mobile broadcaster page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
