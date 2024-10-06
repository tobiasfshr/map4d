"""Dataset pre-processing utils."""


def frame_in_range(frame, frame_ranges):
    if isinstance(frame_ranges[0], list):
        for fr_ra in frame_ranges:
            if fr_ra[0] <= frame <= fr_ra[1]:
                return True
    else:
        return frame_ranges[0] <= frame <= frame_ranges[1]
    return False
