# Contributing to the ALE

We welcome all forms of contributions! Please give the following a read before submitting a PR.

## Pull Requests

1. Fork the repo and create your branch from master.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes. e.g., `cmake --build . --config Release --target test`

## Code Style

If you would like to make changes to the codebase, please adhere to the
following code style conventions.

ALE contains two sets of source files: Files .hxx and .cxx are part of the
Stella emulator code. Files .hpp and .cpp are original ALE code. The Stella
files are not subject to our conventions, please retain their local style.

The ALE code style conventions are roughly summarised as "clang-format with the
following settings: ReflowComments: false, PointerAlignment: Left,
KeepEmptyLinesAtTheStartOfBlocks: false, IndentCaseLabels: true,
AccessModifierOffset: -1". That is:

- Indent by two spaces; Egyptian braces, no extraneous newlines at the margins
  of blocks and between top-level declarations.
- Pointer/ref qualifiers go on the left (e.g. `void* p`).
- Class member access modifiers are indented by _one_ space.
- Inline comments should be separated from code by two spaces (though this is
  not currently applied consistently).
- There is no strict line length limit, but keep it reasonable.
- Namespace close braces and `#endif`s should have comments.

The overall format should look reasonably "compact" without being crowded. Use
blank lines generously _within_ blocks and long comments to create visual cues
for the segmentation of ideas.

## License

By contributing, you agree that your contributions will be licensed under the GPLv2 License.
