# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

### Changed
- Renamed package from `anyimage` to `anybioimage`
- Updated all imports: `from anyimage import ...` → `from anybioimage import ...`
- Updated repository URLs and documentation
- Updated GitHub workflows and CI configuration

### Added
- Automated PyPI publishing via GitHub releases
- Release process documentation (RELEASING.md)
- This changelog

## [0.2.0] - Previous Release

Initial release as `anyimage` with core functionality:
- Multi-dimensional image viewer (5D: TCZYX)
- Mask overlay support
- Annotation tools (rectangles, polygons, points)
- SAM (Segment Anything Model) integration
- HCS plate support for OME-Zarr
- Tile-based rendering with caching
