%This function gets extracts all data from one log folder and saves it as
% mat file in the -hdf5 format.

function getBasicGokartData (logDir = "/home/mvb/0_ETH/MasterThesis/Logs_GoKart/LogData/20190204/20190204T185052_00", filename = "20190204T185052_00.mat")
  gokartData = {};

  cd(logDir);

  dataZip = readdir(logDir);
  saveornot = true;
  tic
  for i = 1:length(dataZip)
%    printf(strcat(dataZip{i,1},'\n'));
    [info, err, msg] = stat (dataZip{i,1});
    if length(dataZip{i,1}) > 2
      if dataZip{i,1}(end) == "z" && dataZip{i,1}(end-1) == "g"
        if info.size > 1000
        %steerTorque
          if typeinfo(strfind (dataZip{i,1}, 'steer')) == 'scalar'
            if typeinfo(strfind (dataZip{i,1}, 'put')) == 'scalar'
              file = load (dataZip{i,1});
              gokartData.steerTorque.command.time = file(:,1);
              gokartData.steerTorque.command.data = file(:,3);
              gokartData.steerTorque.command.title = "commanded";
            endif
          endif
          if typeinfo(strfind (dataZip{i,1}, 'steer')) == 'scalar'
            if typeinfo(strfind (dataZip{i,1}, 'get')) == 'scalar'
              file = load (dataZip{i,1});
            %steer torqque
              gokartData.steerTorque.effective.time = file(:,1);
              gokartData.steerTorque.effective.data = file(:,6);
              gokartData.steerTorque.effective.title = "effective";
            %steer position
              gokartData.steerPosition.raw.time = gokartData.steerTorque.effective.time;
              gokartData.steerPosition.raw.data = file(:,9);
              gokartData.steerPosition.raw.title = "raw";
            endif
          endif
        %steerPosition
          if typeinfo(strfind (dataZip{i,1}, 'status')) == 'scalar'
            if typeinfo(strfind (dataZip{i,1}, 'get')) == 'scalar'
              file = load (dataZip{i,1});
              gokartData.steerPosition.calibrated.time = file(:,1);
              gokartData.steerPosition.calibrated.data = file(:,2);
              gokartData.steerPosition.calibrated.title = "calibrated";
            endif
          endif
        %poseSmooth
          if typeinfo(strfind (dataZip{i,1}, 'pose')) == 'scalar'
            if typeinfo(strfind (dataZip{i,1}, 'smooth')) == 'scalar'
              file = load (dataZip{i,1});
              gokartData.poseSmooth.x.time = file(:,1);
              gokartData.poseSmooth.x.data = file(:,2);
              gokartData.poseSmooth.x.title = "x";
              gokartData.poseSmooth.y.time = gokartData.poseSmooth.x.time;
              gokartData.poseSmooth.y.data = file(:,3);
              gokartData.poseSmooth.y.title = "y";
              gokartData.poseSmooth.heading.time = gokartData.poseSmooth.x.time;
              gokartData.poseSmooth.heading.data = file(:,4);
              gokartData.poseSmooth.heading.title = "heading";
            endif
          endif
        %pose2D
          if typeinfo(strfind (dataZip{i,1}, 'pose')) == 'scalar'
            if typeinfo(strfind (dataZip{i,1}, 'smooth')) == 'scalar'
              file = load (dataZip{i,1});
              gokartData.pose2D.xy.time = file(:,2);
              gokartData.pose2D.xy.data = file(:,3);
              gokartData.pose2D.xy.title = "2D Pose";
            endif
          endif
        %linmotPosition
          if typeinfo(strfind (dataZip{i,1}, 'linmot')) == 'scalar'
            if typeinfo(strfind (dataZip{i,1}, 'put')) == 'scalar'
              file = load (dataZip{i,1});
              gokartData.linmotPosition.command.time = file(:,1);
              gokartData.linmotPosition.command.data = file(:,2);
              gokartData.linmotPosition.command.title = "commanded";
            endif
          endif
          if typeinfo(strfind (dataZip{i,1}, 'linmot')) == 'scalar'
            if typeinfo(strfind (dataZip{i,1}, 'get')) == 'scalar'
              file = load (dataZip{i,1});
              gokartData.linmotPosition.effective.time = file(:,1);
              gokartData.linmotPosition.effective.data = file(:,2);
              gokartData.linmotPosition.effective.title = "effective";
            endif
          endif
        %rimoTorqueCommand
          if typeinfo(strfind (dataZip{i,1}, 'rimo')) == 'scalar'
            if typeinfo(strfind (dataZip{i,1}, 'put')) == 'scalar'
              file = load (dataZip{i,1});
              gokartData.rimoTorqueCommand.left.time = file(:,1);
              gokartData.rimoTorqueCommand.left.data = file(:,2);
              gokartData.rimoTorqueCommand.left.title = "left";
              gokartData.rimoTorqueCommand.right.time = gokartData.rimoTorqueCommand.left.time;
              gokartData.rimoTorqueCommand.right.data = file(:,3);
              gokartData.rimoTorqueCommand.right.title = "right";
            endif
          endif
        %rimoRotationalRate
          if typeinfo(strfind (dataZip{i,1}, 'rimo')) == 'scalar'
            if typeinfo(strfind (dataZip{i,1}, 'get')) == 'scalar'
              file = load (dataZip{i,1});
              gokartData.rimoRotationalRate.left.time = file(:,1);
              gokartData.rimoRotationalRate.left.data = file(:,3);
              gokartData.rimoRotationalRate.left.title = "left";
              gokartData.rimoRotationalRate.right.time = gokartData.rimoRotationalRate.left.time;
              gokartData.rimoRotationalRate.right.data = file(:,10);
              gokartData.rimoRotationalRate.right.title = "right";
            endif
          endif
        %vmu931
          if typeinfo(strfind (dataZip{i,1}, 'vmu931')) == 'scalar'
            if typeinfo(strfind (dataZip{i,1}, 'vehicle')) == 'scalar'
              file = load (dataZip{i,1});
              gokartData.vmu931.xdtdt.time = [file(:,1)];
              gokartData.vmu931.xdtdt.data = [file(:,3)];
              gokartData.vmu931.xdtdt.title = "x_{dotdot} (forward)";
              gokartData.vmu931.ydtdt.time = gokartData.vmu931.xdtdt.time;
              gokartData.vmu931.ydtdt.data = [file(:,4)];
              gokartData.vmu931.ydtdt.title = "y_{dotdot} (left)";
              gokartData.vmu931.headingdt.time = gokartData.vmu931.xdtdt.time;
              gokartData.vmu931.headingdt.data = [file(:,5)];
              gokartData.vmu931.headingdt.title = "rotational rate";
            endif
          endif
        else
          saveornot = false;
          printf(strcat("Extraction of:_",dataZip{i,1} , "_failed!\n\n"));
        endif
      endif
    endif
  endfor
  elapsed_time = toc
  cd ..
  if saveornot
%    printf("Extraction successful!\n\n");
    save('-hdf5', filename, 'gokartData')
  endif
endfunction