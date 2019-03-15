function gokartData = postProcessData (gokartData)

  %poseSmoothdt: Differentiate pose
  gokartData.poseSmoothdt.vx.time = gokartData.poseSmooth.x.time(1:end-1);
  gokartData.poseSmoothdt.vx.data = diff(gokartData.poseSmooth.x.data)./diff(gokartData.poseSmooth.x.time);
  gokartData.poseSmoothdt.vx.title = "v_x (from poseSmooth)";
  gokartData.poseSmoothdt.vy.time = gokartData.poseSmoothdt.vx.time;
  gokartData.poseSmoothdt.vy.data = diff(gokartData.poseSmooth.y.data)./diff(gokartData.poseSmooth.y.time);
  gokartData.poseSmoothdt.vy.title = "v_y (from poseSmooth)";
  dheading = diff(gokartData.poseSmooth.heading.data);
  for j = 1:length(dheading)
    if dheading(j)>pi
      dheading(j) = dheading(j)-2*pi;
    elseif dheading(j)<-pi
      dheading(j) = dheading(j)+2*pi;
    endif
  endfor
  gokartData.poseSmoothdt.headingdt.time = gokartData.poseSmoothdt.vx.time;
  gokartData.poseSmoothdt.headingdt.data = dheading./diff(gokartData.poseSmooth.heading.time);
  gokartData.poseSmoothdt.headingdt.title = "rotational rate";

  sigma = 5;
  sz = 3*sigma;    % length of gaussFilter vector
  x = linspace(-sz / 2, sz / 2, sz);
  gaussFilter = exp(-x .^ 2 / (2 * sigma ^ 2));
  gaussFilter = gaussFilter / sum (gaussFilter); % normalize
  gokartData.poseSmoothdt.vxfilter.time = gokartData.poseSmoothdt.vx.time;
  gokartData.poseSmoothdt.vxfilter.data = conv (gokartData.poseSmoothdt.vx.data, gaussFilter, 'same');
  gokartData.poseSmoothdt.vxfilter.title = "v_x (filtered)";
  gokartData.poseSmoothdt.vyfilter.time = gokartData.poseSmoothdt.vx.time;
  gokartData.poseSmoothdt.vyfilter.data = conv (gokartData.poseSmoothdt.vy.data, gaussFilter, 'same');
  gokartData.poseSmoothdt.vyfilter.title = "v_y (filtered)";
  gokartData.poseSmoothdt.headingdtfilter.time = gokartData.poseSmoothdt.vx.time;
  gokartData.poseSmoothdt.headingdtfilter.data = conv (gokartData.poseSmoothdt.headingdt.data, gaussFilter, 'same');
  gokartData.poseSmoothdt.headingdtfilter.title = "rotational rate (filtered)";
  %poseSmoothdtdt: Differentiate posedt
  gokartData.poseSmoothdtdt.ax.time = gokartData.poseSmoothdt.vx.time(1:end-1);
  gokartData.poseSmoothdtdt.ax.data = diff(gokartData.poseSmoothdt.vxfilter.data)./diff(gokartData.poseSmoothdt.vxfilter.time);
  gokartData.poseSmoothdtdt.ax.title = "a_x (from poseSmooth)";
  gokartData.poseSmoothdtdt.ay.time = gokartData.poseSmoothdtdt.ax.time;
  gokartData.poseSmoothdtdt.ay.data = diff(gokartData.poseSmoothdt.vyfilter.data)./diff(gokartData.poseSmoothdt.vyfilter.time);
  gokartData.poseSmoothdtdt.ay.title = "a_y (from poseSmooth)";
  gokartData.poseSmoothdtdt.headingdtdt.time = gokartData.poseSmoothdtdt.ax.time;
  gokartData.poseSmoothdtdt.headingdtdt.data = diff(gokartData.poseSmoothdt.headingdtfilter.data)./diff(gokartData.poseSmoothdt.headingdtfilter.time);
  gokartData.poseSmoothdtdt.headingdtdt.title = "rotational acceleration (from poseSmooth)";
  
  %poseVehicledt: Differentiate pose wrt heading
  gokartData.vehicleStatedt.xdt.time = gokartData.poseSmooth.x.time(1:end-1);
  %            gokartData.poseVehicledt.heading = gokartData.poseSmooth.heading(1:end-1);
  %            gokartData.poseVehicledt.headingTitle = "heading (from poseSmooth)";
  %            gokartData.poseVehicledt.atan2data = atan2(gokartData.poseSmoothdt.yfilter,gokartData.poseSmoothdt.xfilter);
  %            gokartData.poseVehicledt.atan2dataTitle = "atan2(vy/vx)";
  gokartData.vehicleStatedt.beta.data = gokartData.poseSmooth.heading.data(1:end-1) .- atan2(gokartData.poseSmoothdt.vyfilter.data,gokartData.poseSmoothdt.vxfilter.data);
%  gokartData.vehicleStatedt.beta.data = gokartData.poseSmooth.heading.data(1:end-1) .- atan2(gokartData.poseSmoothdt.vy.data,gokartData.poseSmoothdt.vx.data);
  for l = 1 : length(gokartData.vehicleStatedt.beta.data)
    if gokartData.vehicleStatedt.beta.data(l)>pi
      gokartData.vehicleStatedt.beta.data(l) = gokartData.vehicleStatedt.beta.data(l) - 2*pi;
    elseif gokartData.vehicleStatedt.beta.data(l)<-pi
      gokartData.vehicleStatedt.beta.data(l) = gokartData.vehicleStatedt.beta.data(l) + 2*pi;
    endif
  endfor
  gokartData.vehicleStatedt.beta.time = gokartData.vehicleStatedt.xdt.time;
  gokartData.vehicleStatedt.beta.title = "slip angle (from poseSmooth)";
  cosbeta = cos(gokartData.vehicleStatedt.beta.data);
  sinbeta = sin(gokartData.vehicleStatedt.beta.data);
  gokartData.vehicleStatedt.xdt.data = sqrt(gokartData.poseSmoothdt.vx.data.^2 + gokartData.poseSmoothdt.vy.data.^2).*cosbeta;
  gokartData.vehicleStatedt.xdt.title = "tangential velocity (from poseSmooth)";
  gokartData.vehicleStatedt.ydt.time = gokartData.vehicleStatedt.xdt.time;
  gokartData.vehicleStatedt.ydt.data = sqrt(gokartData.poseSmoothdt.vx.data.^2 + gokartData.poseSmoothdt.vy.data.^2).*sinbeta;
  gokartData.vehicleStatedt.ydt.title = "side slip velocity (from poseSmooth)";
  
  sigma = 5;
  sz = 3*sigma;    % length of gaussFilter vector
  x = linspace(-sz / 2, sz / 2, sz);
  gaussFilter = exp(-x .^ 2 / (2 * sigma ^ 2));
  gaussFilter = gaussFilter / sum (gaussFilter); % normalize
  gokartData.vehicleStatedt.xdtfilter.time = gokartData.vehicleStatedt.xdt.time;
  gokartData.vehicleStatedt.xdtfilter.data = conv (gokartData.vehicleStatedt.xdt.data, gaussFilter, 'same');
  gokartData.vehicleStatedt.xdtfilter.title = "x_{dot} (filtered)";
  gokartData.vehicleStatedt.ydtfilter.time = gokartData.vehicleStatedt.xdt.time;
  gokartData.vehicleStatedt.ydtfilter.data = conv (gokartData.vehicleStatedt.ydt.data, gaussFilter, 'same');
  gokartData.vehicleStatedt.ydtfilter.title = "y_{dot} (filtered)";
  %poseVehicledtdt: Differentiate poseVehicledt
  gokartData.vehicleStatedtdt.xdtdt.time = gokartData.vehicleStatedt.xdt.time(1:end-1);
  gokartData.vehicleStatedtdt.xdtdt.data = diff(gokartData.vehicleStatedt.xdtfilter.data)./diff(gokartData.vehicleStatedt.xdtfilter.time) - gokartData.poseSmoothdt.headingdt.data(1:end-1) .* gokartData.vehicleStatedt.ydt.data(1:end-1);
  gokartData.vehicleStatedtdt.xdtdt.title = "x_{dotdot} (from poseVehicledt)";
  gokartData.vehicleStatedtdt.ydtdt.time = gokartData.vehicleStatedtdt.xdtdt.time;
  gokartData.vehicleStatedtdt.ydtdt.data = diff(gokartData.vehicleStatedt.ydtfilter.data)./diff(gokartData.vehicleStatedt.ydtfilter.time) + gokartData.poseSmoothdt.headingdt.data(1:end-1) .* gokartData.vehicleStatedt.xdt.data(1:end-1);
  gokartData.vehicleStatedtdt.ydtdt.title = "y_{dotdot} (from poseVehicledt)";
  %            gokartData.poseVehicledtdt.xfromaccel = gokartData.poseSmoothdtdt.x .* cos(gokartData.poseSmoothdt.heading(1:end-1)) + gokartData.poseSmoothdtdt.y .* sin(gokartData.poseSmoothdt.heading(1:end-1));
  %            gokartData.poseVehicledtdt.xfromaccelTitel = "x_{dotdot} (from poseSmoothdtdt)";
  %            gokartData.poseVehicledtdt.yfromaccel = gokartData.poseSmoothdtdt.y .* cos(gokartData.poseSmoothdt.heading(1:end-1)) - gokartData.poseSmoothdtdt.x .* sin(gokartData.poseSmoothdt.heading(1:end-1));
  %            gokartData.poseVehicledtdt.yfromaccelTitel = "y_{dotdot} (from poseSmoothdtdt)";
  
  %vmu931
  dt = gokartData.vmu931.xdtdt.time .- [gokartData.vmu931.xdtdt.time(2:end);gokartData.vmu931.xdtdt.time(end)];
  newtimestep = dt<-0.003;
%  newtimestep = gokartData.vmu931.xdtdt.time<0;
%  t_prev = 0;
%  for i = 1:length(gokartData.vmu931.xdtdt.time)
%    dt = gokartData.vmu931.xdtdt.time(i) - t_prev;
%    if dt > 0.05
%      newtimestep(i,1) = true;
%      t_prev = gokartData.vmu931.xdtdt.time(i);
%    else
%      newtimestep(i,1) = false;
%    endif
%  endfor
  length(gokartData.vmu931.xdtdt.time)
  gokartData.vmu931.xdtdt.time = gokartData.vmu931.xdtdt.time(newtimestep);
  length(gokartData.vmu931.xdtdt.time)
  gokartData.vmu931.xdtdt.data = gokartData.vmu931.xdtdt.data(newtimestep);
  gokartData.vmu931.xdtdt.title = "x_{dotdot} (forward)";
  gokartData.vmu931.ydtdt.time = gokartData.vmu931.xdtdt.time;
  gokartData.vmu931.ydtdt.data = gokartData.vmu931.ydtdt.data(newtimestep);
  gokartData.vmu931.ydtdt.title = "y_{dotdot} (left)";
  gokartData.vmu931.headingdt.time = gokartData.vmu931.xdtdt.time;
  gokartData.vmu931.headingdt.data = gokartData.vmu931.headingdt.data(newtimestep);
  gokartData.vmu931.headingdt.title = "rotational rate";
  %filter data
  sigma = 70;
  sz = 3*sigma;    % length of gaussFilter vector
  x = linspace(-sz / 2, sz / 2, sz);
  gaussFilter = exp(-x .^ 2 / (2 * sigma ^ 2));
  gaussFilter = gaussFilter / sum (gaussFilter); % normalize
  
  gokartData.vmu931Filtered.xdtdt.time = gokartData.vmu931.xdtdt.time;
  gokartData.vmu931Filtered.xdtdt.data = conv (gokartData.vmu931.xdtdt.data, gaussFilter, 'same');
  gokartData.vmu931Filtered.xdtdt.title = "x_{dotdot} vmu (filtered)";
  gokartData.vmu931Filtered.ydtdt.time = gokartData.vmu931.xdtdt.time;
  gokartData.vmu931Filtered.ydtdt.data = conv (gokartData.vmu931.ydtdt.data, gaussFilter, 'same');
  gokartData.vmu931Filtered.ydtdt.title = "y_{dotdot} vmu (filtered)";
  gokartData.vmu931Filtered.headingdt.time = gokartData.vmu931.xdtdt.time;
  gokartData.vmu931Filtered.headingdt.data = conv (gokartData.vmu931.headingdt.data, gaussFilter, 'same');
  gokartData.vmu931Filtered.headingdt.title = "rotational rate vmu (filtered)";
  
  %downsample filtered data
  gokartData.vmu931Filtered.xdtdtds.time = gokartData.vmu931Filtered.xdtdt.time(1:3:end);
  gokartData.vmu931Filtered.xdtdtds.data = gokartData.vmu931Filtered.xdtdt.data(1:3:end);
  gokartData.vmu931Filtered.ydtdtds.time = gokartData.vmu931Filtered.ydtdt.time(1:3:end);
  gokartData.vmu931Filtered.ydtdtds.data = gokartData.vmu931Filtered.ydtdt.data(1:3:end);
  gokartData.vmu931Filtered.headingdtds.time = gokartData.vmu931Filtered.headingdt.time(1:3:end);
  gokartData.vmu931Filtered.headingdtds.data = gokartData.vmu931Filtered.headingdt.data(1:3:end);
  
  %integrate: vmu931int
  gokartData.vmu931int.xdt.time = gokartData.vmu931.xdtdt.time;
  gokartData.vmu931int.xdt.data = cumtrapz (gokartData.vmu931int.xdt.time, gokartData.vmu931Filtered.xdtdt.data);
  gokartData.vmu931int.xdt.title = "x_{dot} (forward)";
  gokartData.vmu931int.ydt.time = gokartData.vmu931.xdtdt.time;
  gokartData.vmu931int.ydt.data = cumtrapz (gokartData.vmu931int.ydt.time, gokartData.vmu931Filtered.ydtdt.data);
  gokartData.vmu931int.ydt.title = "v_{dot} (left)";
  gokartData.vmu931int.heading.time = gokartData.vmu931.xdtdt.time;
  gokartData.vmu931int.heading.data = cumtrapz (gokartData.vmu931int.heading.time, gokartData.vmu931Filtered.headingdt.data);
  gokartData.vmu931int.heading.title = "rotational rate";


  gokartData.slipBehaviour.heading.time = gokartData.vehicleStatedt.xdt.time;
  gokartData.slipBehaviour.heading.data = gokartData.poseSmooth.heading.data(1:end-1);
  gokartData.slipBehaviour.heading.title = gokartData.poseSmooth.heading.title;
  gokartData.slipBehaviour.atan2.time = gokartData.vehicleStatedt.xdt.time;
  gokartData.slipBehaviour.atan2.data = atan2(gokartData.poseSmoothdt.vyfilter.data,gokartData.poseSmoothdt.vxfilter.data);
  gokartData.slipBehaviour.atan2.title = "atan2(vy/vx)";
  gokartData.slipBehaviour.beta.time = gokartData.vehicleStatedt.beta.time;
  gokartData.slipBehaviour.beta.data = gokartData.vehicleStatedt.beta.data;
  gokartData.slipBehaviour.beta.title = gokartData.vehicleStatedt.beta.title;

  gokartData.rotrate.vmu931.time = gokartData.vmu931.headingdt.time;
  gokartData.rotrate.vmu931.data = gokartData.vmu931.headingdt.data;
  gokartData.rotrate.vmu931.title = gokartData.vmu931.headingdt.title;
  gokartData.rotrate.poserotrate.time = gokartData.poseSmoothdt.headingdtfilter.time;
  gokartData.rotrate.poserotrate.data = gokartData.poseSmoothdt.headingdtfilter.data;
  gokartData.rotrate.poserotrate.title = gokartData.poseSmoothdt.headingdtfilter.title;

  gokartData.xDotDot.vmu.time = gokartData.vmu931Filtered.xdtdt.time;
  gokartData.xDotDot.vmu.data = gokartData.vmu931Filtered.xdtdt.data;
  gokartData.xDotDot.vmu.title = gokartData.vmu931Filtered.xdtdt.title;
  gokartData.xDotDot.pose.time = gokartData.vehicleStatedtdt.xdtdt.time;
  gokartData.xDotDot.pose.data = gokartData.vehicleStatedtdt.xdtdt.data;
  gokartData.xDotDot.pose.title = gokartData.vehicleStatedtdt.xdtdt.title;
  %  gokartData.xDotDot.fromaccel = gokartData.poseVehicledtdt.xfromaccel;
  %  gokartData.xDotDot.fromaccelTitle = gokartData.poseVehicledtdt.xfromaccelTitel;

  gokartData.yDotDot.vmu.time = gokartData.vmu931Filtered.ydtdt.time;
  gokartData.yDotDot.vmu.data = gokartData.vmu931Filtered.ydtdt.data;
  gokartData.yDotDot.vmu.title = gokartData.vmu931Filtered.ydtdt.title;
  gokartData.yDotDot.vmudownsample.time = gokartData.vmu931Filtered.ydtdtds.time;
  gokartData.yDotDot.vmudownsample.data = gokartData.vmu931Filtered.ydtdtds.data;
  gokartData.yDotDot.vmudownsample.title = gokartData.vmu931Filtered.ydtdt.title;
  gokartData.yDotDot.pose.time = gokartData.vehicleStatedtdt.ydtdt.time;
  gokartData.yDotDot.pose.data = gokartData.vehicleStatedtdt.ydtdt.data;
  gokartData.yDotDot.pose.title = gokartData.vehicleStatedtdt.ydtdt.title;
  %  gokartData.yDotDot.fromaccel = gokartData.poseVehicledtdt.yfromaccel;
  %  gokartData.yDotDot.fromaccelTitle = gokartData.poseVehicledtdt.yfromaccelTitel;
  
  gokartData.noiseanalysis.xdtdt.time = gokartData.vmu931Filtered.xdtdt.time;
  gokartData.noiseanalysis.xdtdt.data = gokartData.vmu931Filtered.xdtdt.data;
  gokartData.noiseanalysis.xdtdt.title = gokartData.vmu931Filtered.xdtdt.title;
  gokartData.noiseanalysis.posex.time = gokartData.vehicleStatedtdt.xdtdt.time;
  gokartData.noiseanalysis.posex.data = gokartData.vehicleStatedtdt.xdtdt.data;
  gokartData.noiseanalysis.posex.title = gokartData.vehicleStatedtdt.xdtdt.title;
  gokartData.noiseanalysis.ydtdt.time = gokartData.vmu931Filtered.ydtdt.time;
  gokartData.noiseanalysis.ydtdt.data = gokartData.vmu931Filtered.ydtdt.data;
  gokartData.noiseanalysis.ydtdt.title = gokartData.vmu931Filtered.ydtdt.title;
  gokartData.noiseanalysis.posey.time = gokartData.vehicleStatedtdt.ydtdt.time;
  gokartData.noiseanalysis.posey.data = gokartData.vehicleStatedtdt.ydtdt.data;
  gokartData.noiseanalysis.posey.title = gokartData.vehicleStatedtdt.ydtdt.title;
  gokartData.noiseanalysis.left.time = gokartData.rimoTorqueCommand.left.time;
  gokartData.noiseanalysis.left.data = gokartData.rimoTorqueCommand.left.data/100.0;
  gokartData.noiseanalysis.left.title = gokartData.rimoTorqueCommand.left.title;
  gokartData.noiseanalysis.right.time = gokartData.rimoTorqueCommand.right.time;
  gokartData.noiseanalysis.right.data = gokartData.rimoTorqueCommand.right.data/100.0;
  gokartData.noiseanalysis.right.title = gokartData.rimoTorqueCommand.right.title;
  
  sniplet = 1;
  step = 5;
  gokartData.slipangleanalysis.position.x = gokartData.poseSmooth.x.data(1:step:end*sniplet);
  gokartData.slipangleanalysis.position.y = gokartData.poseSmooth.y.data(1:step:end*sniplet);
  gokartData.slipangleanalysis.position.title = "position";
  gokartData.slipangleanalysis.heading.data = gokartData.poseSmooth.heading.data(1:step:end*sniplet);
  gokartData.slipangleanalysis.heading.title = "heading";
  gokartData.slipangleanalysis.velocity.data = atan2(gokartData.poseSmoothdt.vyfilter.data,gokartData.poseSmoothdt.vxfilter.data)(1:step:end*sniplet);
  gokartData.slipangleanalysis.velocity.title = "abs. velocity";
  
            
endfunction