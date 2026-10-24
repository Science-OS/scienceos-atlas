"""Central media store: researcher photos, institution logos and paper thumbnails, catalogued once and reused by every client.

Layout (contract v1), rooted at a directory that is the `media` git branch in CI:
    b/<sha256[0:2]>/<sha256>.<webp|jpg>   immutable bytes (only freely licensed images)
    c/<kind>/<key>.json                   catalogue entry per entity key (a 404 means unknown)
    manifest.json                         counts + hosted size
"""
