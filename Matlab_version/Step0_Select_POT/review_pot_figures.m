function review_pot_figures()
% =========================================================
% POT Figure Interactive Reviewer
% ---------------------------------------------------------
% 浏览 POT_Figures 中的图像，逐张判断保留/剔除
%
% 操作方式：
%   → / D       : 下一张
%   ← / A       : 上一张
%   K / ↑       : 标记为【保留】
%   X / ↓       : 标记为【剔除】
%   U           : 撤销当前标记（恢复未标记）
%   G           : 跳转到指定编号
%   S           : 立即保存进度
%   Q           : 退出并保存结果
%
% 输出：
%   kept_dates.txt    保留的日期对
%   rejected_dates.txt 剔除的日期对
%   review_progress.mat 中间进度（可断点续做）
% =========================================================

clear; clc;

% ---- USER SETTINGS -------------------------------------------------------
base_dir   = '/data2/Phd_Work1/ICE_Velocity_Process/GAMMA_POT_Input/Cook_80m';
fig_dir    = fullfile(base_dir, 'POT_Figures2');
range_dir  = fullfile(base_dir, 'range');
azi_dir    = fullfile(base_dir, 'azimuth');
out_dir    = base_dir;   % where to write txt files

% After review: 'copy' or 'move' or 'none' (just write txt)
file_action = 'copy';
% -------------------------------------------------------------------------

% --- Collect figures ------------------------------------------------------
fig_files = dir(fullfile(fig_dir, '*.png'));
if isempty(fig_files)
    error('No PNG files found in %s', fig_dir);
end
n = numel(fig_files);
pair_ids = strrep({fig_files.name}', '.png', '');
fprintf('Found %d figures.\n', n);

% --- Load previous progress if exists -------------------------------------
progress_file = fullfile(out_dir, 'review_progress.mat');
if exist(progress_file, 'file')
    q = input('Found previous progress. Resume? (y/n): ', 's');
    if strcmpi(strtrim(q), 'y')
        tmp = load(progress_file);
        labels = tmp.labels;   % 1=keep, -1=reject, 0=unreviewed
        fprintf('Resumed. %d reviewed so far.\n', sum(labels ~= 0));
    else
        labels = zeros(n, 1);
    end
else
    labels = zeros(n, 1);
end

% Find first unreviewed
start_idx = find(labels == 0, 1);
if isempty(start_idx); start_idx = 1; end

% --- Build UI -------------------------------------------------------------
hfig = uifigure('Name','POT Figure Reviewer','Position',[50 50 1400 900]);
hfig.Color = [0.15 0.15 0.15];

% Image axes
hax = uiaxes(hfig, 'Position',[10 80 1100 800]);
hax.XTick = []; hax.YTick = [];
hax.Box = 'off'; hax.Color = [0.1 0.1 0.1];

% Status panel (right side)
hpanel = uipanel(hfig,'Position',[1120 80 270 800],...
    'BackgroundColor',[0.2 0.2 0.2],'BorderType','none');

% Progress label
hlabel_prog = uilabel(hpanel,'Position',[5 750 260 40],...
    'Text','','FontSize',14,'FontColor',[0.9 0.9 0.9],...
    'BackgroundColor','none','HorizontalAlignment','center');

% Pair ID label
hlabel_id = uilabel(hpanel,'Position',[5 700 260 45],...
    'Text','','FontSize',13,'FontColor',[1 0.85 0.3],...
    'BackgroundColor','none','HorizontalAlignment','center',...
    'WordWrap','on');

% Status label
hlabel_status = uilabel(hpanel,'Position',[5 640 260 50],...
    'Text','未标记','FontSize',18,'FontWeight','bold',...
    'FontColor',[0.7 0.7 0.7],'BackgroundColor','none',...
    'HorizontalAlignment','center');

% Summary counts
hlabel_summary = uilabel(hpanel,'Position',[5 560 260 70],...
    'Text','','FontSize',13,'FontColor',[0.8 0.8 0.8],...
    'BackgroundColor','none','HorizontalAlignment','center',...
    'WordWrap','on');

% Progress bar background
uipanel(hpanel,'Position',[10 530 250 18],...
    'BackgroundColor',[0.35 0.35 0.35],'BorderType','none');
hprogbar = uipanel(hpanel,'Position',[10 530 1 18],...
    'BackgroundColor',[0.2 0.8 0.4],'BorderType','none');

% Buttons
btn_w = 115; btn_h = 45; btn_y = 460;
uibutton(hpanel,'push','Text','◀ 上一张 (A/←)',...
    'Position',[5 btn_y btn_w btn_h],...
    'FontSize',12,'ButtonPushedFcn',@(~,~) step(-1));
uibutton(hpanel,'push','Text','下一张 (D/→) ▶',...
    'Position',[130 btn_y btn_w btn_h],...
    'FontSize',12,'ButtonPushedFcn',@(~,~) step(1));

uibutton(hpanel,'push','Text','✔ 保留 (K/↑)',...
    'Position',[5 btn_y-55 btn_w btn_h],...
    'FontSize',12,'BackgroundColor',[0.2 0.6 0.2],...
    'FontColor','white','ButtonPushedFcn',@(~,~) setlabel(1));
uibutton(hpanel,'push','Text','✘ 剔除 (X/↓)',...
    'Position',[130 btn_y-55 btn_w btn_h],...
    'FontSize',12,'BackgroundColor',[0.65 0.15 0.15],...
    'FontColor','white','ButtonPushedFcn',@(~,~) setlabel(-1));

uibutton(hpanel,'push','Text','↩ 撤销 (U)',...
    'Position',[5 btn_y-110 btn_w btn_h],...
    'FontSize',12,'ButtonPushedFcn',@(~,~) setlabel(0));
uibutton(hpanel,'push','Text','跳转 (G)',...
    'Position',[130 btn_y-110 btn_w btn_h],...
    'FontSize',12,'ButtonPushedFcn',@(~,~) goto_idx());

uibutton(hpanel,'push','Text','💾 保存进度 (S)',...
    'Position',[5 btn_y-165 115 btn_h],...
    'FontSize',12,'ButtonPushedFcn',@(~,~) save_progress());
uibutton(hpanel,'push','Text','✔ 完成并导出 (Q)',...
    'Position',[130 btn_y-165 115 btn_h],...
    'FontSize',12,'BackgroundColor',[0.2 0.4 0.75],...
    'FontColor','white','ButtonPushedFcn',@(~,~) finish());

% Bottom hint bar
hlabel_hint = uilabel(hfig,'Position',[10 10 1380 35],...
    'Text','K/↑=保留   X/↓=剔除   A/←=上一张   D/→=下一张   U=撤销   G=跳转   S=保存   Q=完成导出',...
    'FontSize',12,'FontColor',[0.6 0.6 0.6],...
    'BackgroundColor',[0.12 0.12 0.12],'HorizontalAlignment','center');

% Keyboard callback
hfig.KeyPressFcn = @key_handler;

% --- State ----------------------------------------------------------------
cur_idx = start_idx;
img_cache = containers.Map('KeyType','int32','ValueType','any');

show_image(cur_idx);

% ---- Wait until window closed --------------------------------------------
waitfor(hfig);

% ==========================================================================
%  NESTED FUNCTIONS
% ==========================================================================

    function show_image(idx)
        cur_idx = idx;
        fname = fullfile(fig_dir, fig_files(idx).name);

        % Cache images for speed
        if isKey(img_cache, int32(idx))
            img = img_cache(int32(idx));
        else
            img = imread(fname);
            if img_cache.Count < 30   % keep up to 30 in cache
                img_cache(int32(idx)) = img;
            end
        end

        imshow(img, 'Parent', hax);
        hax.XTick = []; hax.YTick = [];

        % Update labels
        hlabel_prog.Text  = sprintf('%d / %d', idx, n);
        hlabel_id.Text    = strrep(pair_ids{idx}, '-', ' → ');

        lv = labels(idx);
        if lv == 1
            hlabel_status.Text      = '✔  保留';
            hlabel_status.FontColor = [0.3 0.95 0.4];
        elseif lv == -1
            hlabel_status.Text      = '✘  剔除';
            hlabel_status.FontColor = [1 0.35 0.35];
        else
            hlabel_status.Text      = '—  未标记';
            hlabel_status.FontColor = [0.7 0.7 0.7];
        end

        nk = sum(labels ==  1);
        nr = sum(labels == -1);
        nu = sum(labels ==  0);
        hlabel_summary.Text = sprintf('保留: %d\n剔除: %d\n未标记: %d', nk, nr, nu);

        % Progress bar
        pct = (nk + nr) / n;
        hprogbar.Position(3) = max(1, round(250 * pct));
        drawnow limitrate;
    end

    function step(dir)
        new_idx = cur_idx + dir;
        if new_idx < 1; new_idx = 1; end
        if new_idx > n; new_idx = n; end
        show_image(new_idx);
    end

    function setlabel(val)
        labels(cur_idx) = val;
        show_image(cur_idx);
        % Auto-advance after labelling
        if val ~= 0 && cur_idx < n
            step(1);
        end
    end

    function goto_idx()
        answer = inputdlg(sprintf('跳转到编号 (1 - %d):', n), '跳转', 1, {num2str(cur_idx)});
        if isempty(answer); return; end
        idx = round(str2double(answer{1}));
        if isnan(idx) || idx < 1 || idx > n; return; end
        show_image(idx);
    end

    function save_progress()
        save(progress_file, 'labels', 'pair_ids');
        fprintf('Progress saved to %s\n', progress_file);
        uialert(hfig, sprintf('进度已保存\n(%d 已标记)', sum(labels~=0)), '保存成功');
    end

    function finish()
        % Write txt files
        kept_ids     = pair_ids(labels ==  1);
        rejected_ids = pair_ids(labels == -1);

        kept_file = fullfile(out_dir, 'kept_dates.txt');
        rej_file  = fullfile(out_dir, 'rejected_dates.txt');

        fid = fopen(kept_file, 'w');
        for k = 1:numel(kept_ids); fprintf(fid, '%s\n', kept_ids{k}); end
        fclose(fid);

        fid = fopen(rej_file, 'w');
        for k = 1:numel(rejected_ids); fprintf(fid, '%s\n', rejected_ids{k}); end
        fclose(fid);

        fprintf('Kept:     %d  → %s\n', numel(kept_ids),     kept_file);
        fprintf('Rejected: %d  → %s\n', numel(rejected_ids), rej_file);

        % File operations
        if ~strcmp(file_action, 'none')
            apply_file_action(kept_ids, rejected_ids);
        end

        save(progress_file, 'labels', 'pair_ids');

        msg = sprintf('保留: %d 条\n剔除: %d 条\n未标记: %d 条\n\n结果已写入:\n%s\n%s', ...
            numel(kept_ids), numel(rejected_ids), sum(labels==0), kept_file, rej_file);
        uialert(hfig, msg, '导出完成');
    end

    function apply_file_action(kept_ids, rejected_ids)
        % Create subfolders
        kept_range = fullfile(range_dir, 'kept');
        kept_azi   = fullfile(azi_dir,   'kept');
        rej_range  = fullfile(range_dir, 'rejected');
        rej_azi    = fullfile(azi_dir,   'rejected');
        for d = {kept_range, kept_azi, rej_range, rej_azi}
            if ~exist(d{1},'dir'); mkdir(d{1}); end
        end

        move_files(kept_ids,     range_dir, kept_range, '.range.tif');
        move_files(kept_ids,     azi_dir,   kept_azi,   '.azi.tif');
        move_files(rejected_ids, range_dir, rej_range,  '.range.tif');
        move_files(rejected_ids, azi_dir,   rej_azi,    '.azi.tif');
    end

    function move_files(ids, src_dir, dst_dir, ext)
        for k = 1:numel(ids)
            src = fullfile(src_dir, [ids{k}, ext]);
            dst = fullfile(dst_dir, [ids{k}, ext]);
            if ~exist(src,'file'); continue; end
            if strcmp(file_action, 'move')
                movefile(src, dst);
            else
                copyfile(src, dst);
            end
        end
    end

    function key_handler(~, evt)
        switch evt.Key
            case {'rightarrow','d'};  step(1);
            case {'leftarrow','a'};   step(-1);
            case {'uparrow','k'};     setlabel(1);
            case {'downarrow','x'};   setlabel(-1);
            case 'u';                 setlabel(0);
            case 'g';                 goto_idx();
            case 's';                 save_progress();
            case 'q';                 finish();
        end
    end

end
