
function [] = save_table(tab, row_names, col_names, filename)

    rounding = 2;

    fileID = fopen(filename,'w');
        
        fprintf(fileID, '%s \n', strcat('\begin{tabular}{l ', repmat('c', 1, size(tab,2)),'}'));
        fprintf(fileID, '%s \n', '\toprule');
        % column headers
        s = ' ';
        for c = 1:size(tab,2)
            s = strcat(s, ' & ', col_names{c});
        end
        s = strcat(s, ' \\');
        fprintf(fileID, '%s \n', s);
        fprintf(fileID, '%s \n', '\midrule');
        % rows
        for r = 1:size(tab,1)
            s = row_names{r};
            for c = 1:size(tab,2)
                s = strcat(s, ' & ', num2str(round(tab(r,c), rounding)));
            end
            s = strcat(s, ' \\');
            fprintf(fileID, '%s \n', s);
        end
        fprintf(fileID, '%s \n', '\bottomrule');
        fprintf(fileID, '%s \n', '\end{tabular}');
    fclose(fileID);

end
