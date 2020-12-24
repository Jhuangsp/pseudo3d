import os, csv
import pprint
import glob, argparse

pp = pprint.PrettyPrinter(indent = 4)

def rallyMatcher(rally_csvs, set_csv, rall_seg_csv):
    rallys_info = []

    start_frame = {}
    with open(rall_seg_csv, newline='', encoding='utf-8') as seg_f:
        seg = csv.DictReader(seg_f)
        for rally in seg:
            start_frame[rally['Score']] = int(rally['Start'])
    
    hit_table = {}
    shot_table = {}
    with open(set_csv, newline='', encoding='utf-8') as set_f:
        hits = csv.DictReader(set_f)
        for h in hits:
            rally = '{}_{:02d}_{:02d}'.format(set_csv[-5], int(h['roundscore_A']), int(h['roundscore_B']))
            print(rally)
            hit_frame = '{:d}'.format(int(float(h['frame_num'])))
            if rally in hit_table:
                hit_table[rally].append(hit_frame)
                shot_table[rally][hit_frame] = {
                                                'StartX':h['hit_x'],
                                                'StartY':h['hit_y'],
                                                'EndX':h['landing_x'], 
                                                'EndY':h['landing_y']
                                                }
                # print(hit_table)
                # print(shot_table)
            else:
                hit_table[rally] = [hit_frame]
                shot_table[rally] = {hit_frame:
                                        {
                                        'StartX':h['hit_x'],
                                        'StartY':h['hit_y'],
                                        'EndX':h['landing_x'], 
                                        'EndY':h['landing_y']
                                        }
                                    }

    for rally_csv in rally_csvs:
        name = rally_csv.split(os.sep)[-1].split('.')[0]
        print(name)
        with open(rally_csv, newline='', encoding='utf-8') as ral_f:
            ral = csv.DictReader(ral_f)
            rally_info = []
            for i, frame in enumerate(ral):
                if i == 0: continue
                idx_set = str(int(frame['Frame'])+start_frame[name]-1)
                # print(idx_set)
                # print(hit_table[name])
                # os._exit(0)
                rally_info.append({'Frame':idx_set,
                                   'X':frame['X'],
                                   'Y':frame['Y'],
                                   'Visibility':frame['Visibility'],
                                   'Hit':True if idx_set in hit_table[name] else False,
                                   'StartX':shot_table[name][idx_set]['StartX'] if idx_set in hit_table[name] else None,
                                   'StartY':shot_table[name][idx_set]['StartY'] if idx_set in hit_table[name] else None,
                                   'EndX':shot_table[name][idx_set]['EndX'] if idx_set in hit_table[name] else None,
                                   'EndY':shot_table[name][idx_set]['EndY'] if idx_set in hit_table[name] else None,})
            rallys_info.append(rally_info)
    return rallys_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rallys", type=str, help='Folder contain the rally csv files')
    parser.add_argument("--set", type=str, help='Set csv file')
    parser.add_argument("--seg", type=str, help='Rally Segment csv file')
    args = parser.parse_args()

    rally_files = glob.glob(os.path.join(args.rallys, '{}_*.csv'.format(args.set[-5])))
    rally_files.sort()

    info = rallyMatcher(rally_files, args.set, args.seg)
    pp.pprint(info[0])

    for e, f in enumerate(rally_files):
        with open(os.path.join(args.rallys, 'set_'+f.split(os.sep)[-1]), 'w', newline='') as csvfile:
            fieldnames = ['Frame', 'Visibility', 'X', 'Y', 'Hit', 'StartX', 'StartY', 'EndX', 'EndY']

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in info[e]:
                writer.writerow(row)