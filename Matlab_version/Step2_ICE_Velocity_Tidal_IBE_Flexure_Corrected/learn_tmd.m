clc;clear;close all
% Define a time array:
t = datetime('2017-04-11'):hours(1):datetime('2017-05-23');


% Predict the tide time series:
z = tmd_predict('CATS2008_v2023.nc',-67.2494426,145.3523721,t);

figure
plot(t,z)
box off
ylabel('tide height (m)')


%% part2
