close all
clear all
starttime = time();
function [subDirsNames] = GetSubDirsFirstLevelOnly(parentDir)
    % Get a list of all files and folders in this folder.
    files = dir(parentDir);
    % Get a logical vector that tells which is a directory.
    dirFlags = [files.isdir];
    % Extract only those that are directories.
    subDirs = files(dirFlags);
    subDirsNames = cell(1, numel(subDirs) - 2);
    for i=3:numel(subDirs)
        subDirsNames{i-2} = subDirs(i).name;
    endfor
endfunction

logDayDir = pwd;
foldernames = GetSubDirsFirstLevelOnly(logDayDir);
confirm_recursive_rmdir (false)
for m = 1:length(foldernames)
  logDir = strcat(logDayDir, '/' ,foldernames{m});
  filename = strcat(foldernames{m},'.mat');
  printf(strcat("extracting:_", filename,"\n"))
  getBasicGokartData (logDir, filename);
  rmdir (foldernames{m}, "s");
endfor
overall_extration_time = time() - starttime
