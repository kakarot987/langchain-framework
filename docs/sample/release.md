# Release Process

A maintainer reviews passing CI checks and the retrieval evaluation report before approving a release. All changes first land via reviewed pull requests.

## Release checklist

1. Run Ruff and pytest.
2. Run `docqa evaluate` on the labeled retrieval fixture.
3. Build the Docker image and run the offline demo.
4. Verify the README quickstart on a clean machine.
5. Review the known limitations and update the changelog.

## Rollback

If a release introduces a regression, revert the offending commit through a new reviewed pull request and rebuild the image. This demo has no database or persisted index that needs migration.
