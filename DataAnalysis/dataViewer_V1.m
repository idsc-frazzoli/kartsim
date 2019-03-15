## 20.03.2017 Andreas Weber <andy@josoansi.de>
## Demo which has the aim to show all available GUI elements.
## Useful since Octave 4.0
close all
clear all


function update_plot (obj, init = false, gokartData)
  
  colors = ['blue';'red';'green';'cyan';'magenta';'yellow';'blue';'red';'green';'cyan';'magenta';'black'];
  ## gcbo holds the handle of the control
  linewidth = 2;
  h = guidata (obj);
  replot = false;
  special_replot = false;

  switch (gcbo)
    case {h.grid_checkbox}
      v = get (gcbo, "value");
      grid (merge (v, "on", "off"));
    case {h.minor_grid_toggle}
      v = get (gcbo, "value");
      g = get(h.grid_checkbox, "value");
      if g
        grid ("minor", merge (v, "on", "off"));
      endif
    case {h.plotData_popup}
      get(h.plotData_popup, "value")
      if (get(h.plotData_popup, "value") == 20)
        special_replot = true;
      else
        replot = true;
      endif
%    case {h.logFile_popup}
%      init = true;
  endswitch
  
  if (init)
    plotDataName = 'gokartData.pose2D';
    plotVariables = fieldnames(eval(plotDataName));
    for iter = 1:length(plotVariables)
      x = eval(strcat(plotDataName, '.',plotVariables{iter},'.time'));
      y = eval(strcat(plotDataName, '.',plotVariables{iter},'.data'));
      title = strcat(';', plotVariables{iter}, ';');
      h.plot = plot (x, y, 'color', colors(iter), title, "linewidth",linewidth);
      hold on
      guidata (obj, h);
    endfor
    set (get (h.ax, "title"), "string", "pose2D", "fontsize", 15);
    guidata (obj, h);
  endif
  
  if (replot)
    cla
    set(h.grid_checkbox, "value", 0);
    set(h.minor_grid_toggle, "value", 0);
    plotData = get (h.plotData_popup, "string")(get (h.plotData_popup, "value"),:);
    plotDataName = strcat('gokartData.', plotData);
    plotVariables = fieldnames(eval(plotDataName));
    for iter = 1:length(plotVariables)
      x = eval(strcat(plotDataName, '.',plotVariables{iter},'.time'));
      y = eval(strcat(plotDataName, '.',plotVariables{iter},'.data'));
      title = strcat(';', plotVariables{iter}, ';');
      h.plot = plot (x, y, 'color', colors(iter), title, "linewidth",linewidth);
      hold on
      guidata (obj, h);
    endfor
    hold off
    set (get (h.ax, "title"), "string", plotData, "fontsize", 15);
  endif
  
  if (special_replot)
    cla
    set(h.grid_checkbox, "value", 0);
    set(h.minor_grid_toggle, "value", 0);
    arrowscale = 0.2;
    plotData = get (h.plotData_popup, "string")(get (h.plotData_popup, "value"),:);
    plotDataName = strcat('gokartData.', plotData);
    plotVariables = fieldnames(eval(plotDataName));
    x = eval(strcat(plotDataName, '.',plotVariables{1},'.x'));
    y = eval(strcat(plotDataName, '.',plotVariables{1},'.y'));
    heading = eval(strcat(plotDataName, '.',plotVariables{2},'.data'));
    velocity = eval(strcat(plotDataName, '.',plotVariables{3},'.data'));
    xytitle = strcat(';', plotVariables{1}, ';');
    headingtitle = strcat(';', plotVariables{2}, ';');
    velocitytitle = strcat(';', plotVariables{3}, ';');
    plot(x,y,"r", "linewidth",linewidth)%, xytitle) 
    hold on; 
    quiver (x, y, cos(heading), sin(heading), arrowscale, 'b')%, "test");
    length(x) 
    length(y) 
    length(velocity)
    quiver (x, y, cos(velocity), sin(velocity), arrowscale ,'g')%, velocitytitle);
    hold off;
    set (get (h.ax, "title"), "string", plotData, "fontsize", 15);
    legend("position", "heading", "abs. velocity")
  endif
endfunction

function [gokartData, h] = getNewLogFile(objj)

  logNr = 0;

  hobj = guidata (objj);
  if length(fieldnames(hobj)) > 1
    delete(hobj.ax)
  endif
  hsrc = gcbo ();
  if length(hsrc) != 0
    logNr = (get (hsrc, "value")-1);
  endif
  
%  close all

  graphics_toolkit qt

  hobj.f = figure (1, 'position', [300 1300 800 600]);
  hobj.ax = axes ("position", [0.05 0.3 0.9 0.6]);

%  cd ..
  logDayDir = pwd;
  files = readdir(logDayDir);
  matfiles = strfind (files, 'mat');

  logFiles = 0;
  filenames = num2str([]);
  for i=1:length(matfiles)
    if typeinfo(matfiles{i}) == 'scalar'
      filenames = [filenames;files{i}];
      logFiles += 1;
    endif
  endfor

  lognrstr = ["0"];
  for i = 1:logFiles-1
    lognrstr = [lognrstr;int2str(i)];
  endfor
  
  
  load(filenames(logNr+1,:));
  printf(strcat(filenames(logNr+1,:), " loaded.\n"))
  
  gokartData = postProcessData(gokartData);
  printf("Data postprocessed.\n")
  
  datanames = fieldnames(gokartData);
  datanamesstr = {};
  for i = 1:length(datanames)
    datanamesstr(1,end+1) = datanames{i};
  endfor

  %gokartData = {};
  ## grid
  hobj.grid_checkbox = uicontrol ("style", "checkbox",
                               "units", "normalized",
                               "string", "show grid",
                               "value", 0,
                               "callback", @update_plot,
                               "position", [0.4 0.12 0.35 0.09]);

  hobj.minor_grid_toggle = uicontrol ("style", "togglebutton",
                                   "units", "normalized",
                                   "string", "fine grid",
                                   "callback", @update_plot,
                                   "value", 0,
                                   "position", [0.4 0.05 0.18 0.09]);


  ## choose data to plot
  hobj.plotData_label = uicontrol ("style", "text",
                                 "units", "normalized",
                                 "string", "Data:",
                                 "horizontalalignment", "left",
                                 "position", [0.05 0.12 0.35 0.08]);

  hobj.plotData_popup = uicontrol ("style", "popupmenu",
                                 "units", "normalized",
                                 "string", datanamesstr,
                                 "callback", {@update_plot, gokartData},
                                 "position", [0.05 0.05 0.3 0.06]);
                                 
  ## choose LogFile
  hobj.logFile_label = uicontrol ("style", "text",
                                 "units", "normalized",
                                 "string", "Log file:",
                                 "horizontalalignment", "left",
                                 "position", [0.7 0.12 0.35 0.08]);
  %b = ["pose2D";"poseSmooth"]
  hobj.logFile_popup = uicontrol ("style", "popupmenu",
                                 "units", "normalized",
                                 "string", lognrstr,
                                 "value", logNr+1,
                                 "callback", {@getNewLogFile, gcf},
                                 "position", [0.7 0.05 0.3 0.06]);

                                 
  set (gcf, "color", get(0, "defaultuicontrolbackgroundcolor"))
  guidata (gcf, hobj)
  
  update_plot (gcf, true, gokartData);
  set(hobj.f, "toolbar", "figure")
endfunction

hinit.f = figure (1, 'position', [300 1300 800 600]);
set (gcf, "color", get(0, "defaultuicontrolbackgroundcolor"))
guidata (gcf, hinit)

[gokartData, h] = getNewLogFile(gcf);