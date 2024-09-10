% code for 
% "Progressive keypoint localization and refinement in image matching"
% by F. Bellavia, L. Morelli, C. Colombo e F. Remondino
% *** use at your risk! ***
% - run on linux (otherwise you need to modify the system call for table generation, e.g. pdfcrop, gs, etc.)
% - if you want to run the matching pipelines instead for working with the data provided 
%   you need to setup a python environment
%   (see the variable python_env_activate inside the code of the matcher folder) and install some package
% - the whole run can take a while (due to FaST_Match, you can try to bypass it in the loops)
% - a better/commented code is planned to be released
% maintainer: Fabio Bellavia (fabio.bellavia@unipa.it)

disp('please see the comment on this script')
% get results for table 1 experiments
go
% get results for table 2 experiments
go_next
% generate report.pdf (same values as report_other.pdf, but wrt report_other.pdf histogram bars are normalized by the maximum overall px error)
get_table
% generate report_other.pdf (same values as report.pdf, but wrt report.pdf histogram bars are normalized by the noise offset magnitude)
get_table_other
% generate report_subpix.pdf (value reported are the percentages of matches with error less than 1 px)
get_table_subpix
