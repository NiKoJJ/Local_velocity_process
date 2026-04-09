%% setup.m
%  add toolboxs and dataset to MATLAB path

%% root path
clear; clc;
fprintf('====== prepare enviroment ====== \n');

toolboxs_path = '/data2/Phd_Work1/Cook/Tide_IBE_Correction/Toolboxs/';
tide_model_path= '/data2/Phd_Work1/Cook/Tide_IBE_Correction/Tide_model_Data';

%% add
% toolboxs
fprintf('\n add tools to MATLAB path...\n');
addpath(genpath(fullfile(toolboxs_path, 'Antarctic-Mapping-Tools')));
addpath(genpath(fullfile(toolboxs_path, 'BedMachine')));
addpath(genpath(fullfile(toolboxs_path, 'bedmap3')));
addpath(genpath(fullfile(toolboxs_path, 'Tide-Model-Driver-3')));
addpath(genpath(fullfile(toolboxs_path, 'cmocean')));
addpath(genpath(fullfile(toolboxs_path, 'm_map_v1.4')));
addpath(genpath(fullfile(toolboxs_path, 'CDT')));

% tide model
addpath(genpath(fullfile(tide_model_path, 'CATS2008')));
addpath(genpath(fullfile(tide_model_path, 'CATS2008-v2023')));

%% check
fprintf(' Check Tools ...\n');


if exist('tmd_predict', 'file') == 2
    fprintf('  √ TMD loading successful \n');
else
    warning('  X TMD loading failed ');
end

fprintf('\n============ Finish ============\n');
