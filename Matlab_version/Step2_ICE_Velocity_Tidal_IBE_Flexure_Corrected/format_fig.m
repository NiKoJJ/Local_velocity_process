function format_fig(fig, out_dir, filename)
    % Set all text elements in figure to font size 24 and save as PNG+FIG
    set(findall(fig, '-property', 'FontSize'), 'FontSize', 24);

    % Tighten layout after font resize to prevent label clipping
    set(findall(fig, 'Type', 'axes'), 'TickLength', [0.015 0.025]);

    drawnow;

    if ~exist(out_dir, 'dir'); mkdir(out_dir); end

    exportgraphics(fig, fullfile(out_dir, [filename, '.png']), ...
                   'Resolution', 300, 'BackgroundColor', 'white');
    % savefig(fig, fullfile(out_dir, [filename, '.fig']));

    fprintf('  Figure saved: %s\n', filename);
end