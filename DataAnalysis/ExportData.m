close all
clear all

filename = "20190304T181143_04.mat";

load(filename);

gokartData = postProcessData(gokartData);

%M = [gokartData.poseSmoothdtdt.ax.time(2:end)-gokartData.poseSmoothdtdt.ax.time(1:end-1),...
%      gokartData.poseSmoothdtdt.ax.data(1:end-1),...
%      gokartData.poseSmoothdtdt.ay.data(1:end-1),...
%      gokartData.poseSmoothdtdt.headingdtdt.data(1:end-1)];
l = length(gokartData.poseSmoothdtdt.ax.time) 
M = [gokartData.poseSmoothdtdt.ax.time,...
      gokartData.poseSmoothdtdt.ax.data,...
      gokartData.poseSmoothdtdt.ay.data,...
      gokartData.poseSmoothdtdt.headingdtdt.data,zeros(l,1)];

M = [size(M),gokartData.poseSmooth.heading.data(1),gokartData.poseSmooth.x.data(1),gokartData.poseSmooth.y.data(1);M];
M(1:30,:)

dlmwrite ([filename(1:end-4),"_accelerations"], M, " ")