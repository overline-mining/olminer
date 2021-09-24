cmake_policy(SET CMP0015 NEW)

install(FILES
  config.conf  control  h-config.sh  h-manifest.conf  h-run.sh  h-stats.sh  olminer.sh
  TYPE DATA
  PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
              GROUP_EXECUTE GROUP_READ
              WORLD_EXECUTE WORLD_READ
)
