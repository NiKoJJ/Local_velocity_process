function labels = assign_groups(dates, mode)
    N = length(dates);
    labels = cell(N, 1);

    switch mode

        case 'monthly'
            for i = 1:N
                labels{i} = datestr(dates(i), 'yyyy-mm');
            end

        case 'seasonal'
            % 南极四季（与北半球相反）
            % Summer: 12,1,2  夏季跨年：12月归当年，1-2月归上一年
            % Autumn: 3,4,5   秋季
            % Winter: 6,7,8   冬季
            % Spring: 9,10,11 春季
            for i = 1:N
                m = month(dates(i));
                y = year(dates(i));
                if m == 12
                    labels{i} = sprintf('%d-%d-Summer', y, y+1);
                elseif m == 1 || m == 2
                    labels{i} = sprintf('%d-%d-Summer', y-1, y);
                elseif m >= 3 && m <= 5
                    labels{i} = sprintf('%d-Autumn', y);
                elseif m >= 6 && m <= 8
                    labels{i} = sprintf('%d-Winter', y);
                else   % 9,10,11
                    labels{i} = sprintf('%d-Spring', y);
                end
            end

        case 'yearly'
            for i = 1:N
                labels{i} = sprintf('%d', year(dates(i)));
            end

        otherwise
            win_days = str2double(strrep(mode, 'fixed_', ''));
            if isnan(win_days) || win_days <= 0
                error('未知模式: %s\n可选: monthly/seasonal/yearly/fixed_6/fixed_12/fixed_18/fixed_30/fixed_60', mode);
            end
            t0 = min(dates);
            for i = 1:N
                bin       = floor(days(dates(i) - t0) / win_days);
                bin_start = t0 + days(bin * win_days);
                labels{i} = datetime(bin_start, 'yyyy-mm-dd');
            end
    end
end
