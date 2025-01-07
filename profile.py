import pstats
from pstats import SortKey

# Load the profiling data
stats = pstats.Stats('profile_output.prof')

# Strip unnecessary directories and sort by cumulative time
stats.strip_dirs().sort_stats(SortKey.CUMULATIVE)

# Print the top 50 functions to inspect naming
stats.print_stats(50)
