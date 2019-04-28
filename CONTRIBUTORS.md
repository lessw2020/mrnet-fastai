# INSTRUCTIONS FOR CONTRIBUTORS

If you would like to contribute to this project, please read the guidelines
below before you begin. Once you have read and understand the guidelines, you
should fork the project in GitHub and create a branch for your contribution.
Following the guidelines will reduce the overhead for accepting your pull
request (PR) and, thus, increase the likelihood of it being merged.

## Guidelines

### Coding style
Standard coding style guidelines for Python apply. Please see the
[PEP-8 Style Guide](https://www.python.org/dev/peps/pep-0008/).

### General considerations
- Contributions will typically be made in the form of changes to existing code
(fixes or patches) or new additions to the project (features).
- When making a contribution, try to be reductionist in your approach. Creating
a separate branch (and, thus, a separate PR) for each file you are changing or
each feature you are adding will make it easier for the maintainers to review
your PR.
- Commit messages should be descriptive, but succinct. They should be written in
the present tense, active voice (e.g. "fix load_data function").
- If you would like to add a new feature, please first review the "TO-DO LIST"
table below to see if there is a desired feature that does not yet have an
assigned contributor. If one exists that you believe you can help with, please
email the maintainers and ask them to assign you to the feature. Then proceed
as below.

### Fixes to existing code
1. Update your local master to keep it current with the upstream master.
2. Name your branch according to the following convention:
`{username}-patch-{number}` (e.g. `wfwiggins-patch-1`)
3. Make as few changes as possible to achieve your goal.

### New features
1. Update your local master to keep it current with the upstream master.
2. Name your branch according to the following convention:
`feature-{feature_name}` (e.g. `feature-contrib`)
3. If you are extending an existing Jupyter notebook, please first make a local
copy of the notebook and name it with either of the following conventions:
`old_notebook_extended.ipynb` or `old_notebook_{initials}.ipynb`.
4. If your feature depends on code in an existing Jupyter notebook, but is not
well represented in the title of the notebook, please create a new notebook and
either import or copy the needed code from the original notebook.
5. If you are adding to the code base. Please add only one feature per
branch/PR. Please also attempt to keep the number of files small and attempt to
avoid changes to existing code.
6. If changes to existing code are absolutely necessary, please add a comment
within the code to justify the change. It is better to add arguments to existing
functions or create new functions that extend the capabilities of existing
functions, rather than to make changes that could potentially break other
features that depend on the existing code.

## TO-DO LIST
<table>
  <tr>
    <th>Feature</th>
    <th>Category</th>
    <th>Assignee(s)</th>
    <th>Status</th>
    <th>Notes</th>
  </tr>
  <tr>
    <td>Data visualization</td>
    <td>EDA</td>
    <td>Walter</td>
    <td>complete</td>
    <td></td>
  </tr>
  <tr>
    <td>Data distribution</td>
    <td>EDA</td>
    <td>Nick</td>
    <td>dev</td>
    <td></td>
  </tr>
  <tr>
    <td>Activation function selection</td>
    <td>Model architecture</td>
    <td>Less</td>
    <td>dev</td>
    <td></td>
  </tr>
  <tr>
    <td>Pixel distribution checkpoint</td>
    <td>Preprocessing</td>
    <td>Walter</td>
    <td>dev</td>
    <td></td>
  </tr>
  <tr>
    <td>Basic arch for "abnormal" task</td>
    <td>Model architecture</td>
    <td></td>
    <td>unassigned</td>
    <td>XResNet50?</td>
  </tr>
</table>
