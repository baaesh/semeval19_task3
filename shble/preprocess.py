import emoji

def remove_duplicated_emojis(lines):
    lines_e = []
    for line in lines:
        line_list = list(line)
        for i in range(len(line_list)):
            if line_list[i] in emoji.UNICODE_EMOJI and line_list[i] == line_list[i + 1]:
                line_list[i] = ''
        line_e = ''.join(line_list)
        lines_e.append(line_e)
    return lines_e
