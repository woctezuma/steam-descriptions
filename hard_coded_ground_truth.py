import steamspypi


def get_app_ids_which_app_name_contains(name_str='Half-Life'):
    data_request = dict()
    data_request['request'] = 'all'

    data = steamspypi.download(data_request)

    app_ids = sorted([data[d]['appid'] for d in data.keys()
                      if name_str in data[d]['name']])

    return app_ids


def get_retrieval_ground_truth():
    # Lists of appIDs:
    # - first obtained with calls to get_app_ids_which_app_name_contains()
    # - then refined manually

    retrieval_ground_truth_as_list = [
        [20, 440, 943490],  # Team Fortress
        [10, 80, 240, 730, 273110, 100],  # Counter-Strike
        [500, 550],  # Left 4 Dead
        [50, 70, 130, 220, 280, 320, 340, 360, 380, 420, 466270, 723390],  # Half-Life
        [24240, 218620],  # PAYDAY
        [400, 620, 104600, 247120, 684410, 659, 52003, 450390],  # Portal
        [12100, 12110, 12120, 12170, 12180, 12210, 12220, 271590],  # Grand Theft Auto
        [22320, 22330, 72850, 306130, 364470, 489830, 611670],  # The Elder Scrolls

        [3900, 3910, 3920, 7600, 8930, 16810, 65980, 244070, 244090, 282210, 289070, 327380, 327390, 327400, 50100,
         34470, 34440, 34450, 3990],  # Sid Meier's Civilization

        [8955, 8980, 49520, 261640, 330830],  # Borderlands
        [219740, 322330],  # Don't Starve
        [15700, 15710, 15740, 15750, 314660],  # Oddworld: Abe
        [236870, 863550, 6850, 6860, 6900, 203140, 205930, 247430, 427820],  # HITMAN
        [346110, 407530, 529910],  # ARK

        [375180, 709010, 660120, 324760, 270880, 751660, 935730, 900020, 320310, 1020600, 285500, 258760, 232010,
         932300, 273740, 494670, 451660, 601170, 273750, 286810, 273760, 227300, 374120, 302060, 286830, 847870, 601590,
         889470],  # Euro Truck Simulator

        [319630, 532210, 554620],  # Life is Strange

        [7000, 8000, 8140, 203160, 224960, 224980, 225000, 225020, 225300, 225320, 233410, 391220,
         750920],  # Tomb Raider

        [21600, 61500, 61510, 61520, 105450, 217750, 221380, 226840, 230070, 264120, 266840, 314970, 341150, 351480,
         362740, 369080, 371710, 397770, 402880, 421060, 431700, 442500, 454600, 556300, 570970, 586080, 597970, 599060,
         601520, 603850, 639300, 678970, 718850, 725870, 783590, 792930, 799890, 817390, 832770, 882110, 882410, 988480,
         997480],  # Age of Empires

        [12530, 12690, 253710, 290730, 322920, 323240, 328670, 328940, 361370, 459940, 518790, 545920, 545930, 545940,
         580930, 585080, 619330, 679190, 758470, 801080, 806230, 860670, 934550, 988340, 1029380, 437210, 455700,
         500140, 513680],  # theHunter

        [20900, 20920, 292030, 303800, 499450, 973760, 544750],  # The Witcher
        [22300, 22370, 22380, 22490, 38400, 38410, 38420, 377160, 588430, 611660],  # Fallout

        [2620, 2630, 2640, 3020, 6810, 7940, 10090, 10180, 21980, 22340, 41700, 42700, 202970, 209160, 209650, 214630,
         251390, 270130, 292730, 311210, 336060, 350330, 358360, 359620, 389470, 390660, 393080, 399810, 476600, 518790,
         626630, 630670, 672680, 765770, 836260, 896840, 987790],  # Call of Duty

        [73010, 225420, 231140, 255710, 261940, 313010, 446010, 457600, 520680, 708280, 845440, 862110, 872730,
         24780],  # Cities

        [57300, 239200, 359390],  # Amnesia
        [1250, 232090, 326960, 690810],  # Killing Floor
        [7670, 8850, 8870, 409710, 409720],  # BioShock
        [209080, 442080, 608800, 49800],  # Guns of Icarus
        [9480, 55230, 206420, 301910],  # Saints Row
    ]

    # Create a dictionary

    retrieval_ground_truth = dict()
    for cluster in retrieval_ground_truth_as_list:
        for element in cluster:
            retrieval_ground_truth[element] = set(cluster) - {element}

    return retrieval_ground_truth


def compute_retrieval_score(query_app_ids, reference_app_id_counters, num_elements_displayed=10, verbose=True):
    retrieval_ground_truth = get_retrieval_ground_truth()

    retrieval_score = 0

    for query_counter, query_app_id in enumerate(query_app_ids):
        reference_app_id_counter = reference_app_id_counters[query_counter]

        try:
            current_retrieval_ground_truth = retrieval_ground_truth[query_app_id]
        except KeyError:
            continue

        current_retrieval_score = 0
        for rank, app_id in enumerate(reference_app_id_counter):

            if app_id in current_retrieval_ground_truth:
                current_retrieval_score += 1

            if rank >= (num_elements_displayed - 1):
                retrieval_score += current_retrieval_score
                if verbose:
                    print('[appID={}] retrieval score = {}'.format(query_app_id, current_retrieval_score))
                break

    print('\nTotal retrieval score = {}'.format(retrieval_score))

    return retrieval_score


if __name__ == '__main__':
    app_ids = get_app_ids_which_app_name_contains(name_str='Half-Life')
    print(app_ids)
