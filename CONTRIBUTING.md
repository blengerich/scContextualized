# Contributing to scContextualized
Thank you for contributing to scContextualized!


## Pull Requests
We welcome your pull requests (PRs).
For minor fixes (e.g., documentation improvements), feel free to submit a PR directly.
If you would like to implement a new feature or a bug, please make sure you (or someone else) has opened an appropriate [issue](https://github.com/blengerich/scContextualized/issues) first; in your PR, please mention the issue it addresses.

### Creating a Pull Request
1. [Fork](https://github.com/blengerich/scContextualized/fork) this repository.
2. Make your code changes locally.
3. Check the style using pylint and black:
    - install pylint-badge: `pip install git+https://github.com/blengerich/pylint-badge`
    - `pylint-badge scContextualized pylint.svg` calculates the current pylint score.
    - `black --check scContextualized --target-version=py38` gives a list of files to be updated by black.
    - Picking a particular file `foo.py` from the list above, I run black `scContextualized/foo.py --target-version=py38`. This makes formatting changes to align with the Black style.
    - Manually inspect the changes with `git diff scContextualized/foo.py`.
    - Re-run `pylint-badge scContextualized pyling.svg` to re-calculate the pylint score, and re-create a badge with that score. Sanity check that pylint score did not decrease by making style changes with black.
4. (Optional) Include your name in alphabetical order in [ACKNOWLEDGEMENTS.md](https://github.com/blengerich/scContextualized/blob/main/ACKNOWLEDGEMENTS.md).
5. Issue a PR to merge your changes.


## Issues
We use GitHub issues to track bugs and feature requests.
Before submitting an issue, please make sure:

1. You have read the README and your question is NOT addressed there.
2. You have done your best to ensure that your issue is NOT a duplicate of one of [the previous issues](https://github.com/blengerich/scContextualized/issues).
3. Your issue is either a bug (unexpected/undesirable behavior) or a feature request.
If it is just a question, please ask it in the [Discussions](https://github.com/blengerich/scContextualized/discussions) forum.

When submitting an issue, please make sure to use the appropriate template.


## License
By contributing to scContextualized, you agree that your contributions will be licensed
under the LICENSE file in the root directory of the source tree.