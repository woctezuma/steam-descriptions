import steamspypi

from utils import load_game_names


def get_top_100_app_ids():
    # Reference: https://github.com/woctezuma/download-steam-banners/blob/master/retrieve_similar_features.py

    data_request = {}
    data_request['request'] = 'top100in2weeks'

    data = steamspypi.download(data_request)

    top_100_app_ids = [int(app_id) for app_id in data]

    return top_100_app_ids


def load_benchmarked_app_ids(append_hard_coded_app_ids=True):
    top_100_app_ids = get_top_100_app_ids()

    # Append hard-coded appIDs

    additional_app_ids_as_str = [
        '620',
        '364470',
        '504230',
        '583950',
        '646570',
        '863550',
        '794600',
        '814380',
    ]
    additional_app_ids = [int(app_id) for app_id in additional_app_ids_as_str]

    benchmarked_app_ids = top_100_app_ids
    if append_hard_coded_app_ids:
        for app_id in set(additional_app_ids).difference(top_100_app_ids):
            benchmarked_app_ids.append(app_id)

    # Remove one hard-coded appID (323370: TERA, not available in my region, so I cannot access the game description)

    removed_app_ids = [323370]

    for app_id in removed_app_ids:
        try:
            index_to_remove = benchmarked_app_ids.index(app_id)
            del benchmarked_app_ids[index_to_remove]
        except ValueError:
            pass

    return benchmarked_app_ids


def get_app_name(app_id, game_names=None):
    if game_names is None:
        game_names, _ = load_game_names(include_genres=False, include_categories=False)

    try:
        app_name = game_names[str(app_id)]
    except KeyError:
        app_name = 'Unknown'

    return app_name


def get_store_url(app_id):
    # Reference: https://github.com/woctezuma/download-steam-banners/blob/master/retrieve_similar_banners.py
    store_url = 'https://store.steampowered.com/app/' + str(app_id)
    return store_url


def get_banner_url(app_id):
    # Reference: https://github.com/woctezuma/download-steam-banners/blob/master/retrieve_similar_features.py
    banner_url = (
        'https://steamcdn-a.akamaihd.net/steam/apps/' + str(app_id) + '/header.jpg'
    )
    return banner_url


def print_ranking(
    query_app_ids,
    reference_app_id_counters,
    num_elements_displayed=10,
    only_print_banners=False,
    game_names=None,
):
    # Reference: https://github.com/woctezuma/download-steam-banners/blob/master/retrieve_similar_features.py

    if game_names is None:
        game_names, _ = load_game_names(include_genres=False, include_categories=False)

    for query_counter, query_app_id in enumerate(query_app_ids):
        app_name = get_app_name(query_app_id, game_names=game_names)

        print(
            '\nQuery appID: {} ([{}]({}))\n'.format(
                query_app_id,
                app_name,
                get_store_url(query_app_id),
            ),
        )

        # Markdown
        # Reference: https://stackoverflow.com/a/14747656
        image_link_str = '[<img alt="{}" src="{}" width="{}">]({})'
        image_width = 150

        reference_app_id_counter = reference_app_id_counters[query_counter]

        for rank, app_id in enumerate(reference_app_id_counter):
            app_name = get_app_name(app_id, game_names=game_names)
            if only_print_banners:
                # Markdown
                print(
                    image_link_str.format(
                        app_name,
                        get_banner_url(app_id),
                        image_width,
                        get_store_url(app_id),
                    ),
                )
            else:
                # No banner, so that this is easier to read in Python console.
                print(
                    '{:2}) app: {} ({} @ {})'.format(
                        rank + 1,
                        app_id,
                        app_name,
                        get_store_url(app_id),
                    ),
                )

            if rank >= (num_elements_displayed - 1):
                break

    return


if __name__ == '__main__':
    benchmarked_app_ids = load_benchmarked_app_ids(append_hard_coded_app_ids=True)
