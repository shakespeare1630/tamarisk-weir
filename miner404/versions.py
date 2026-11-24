from .pipeline import Draft, StagePlan, Step

ACTIVE_PLAN = StagePlan(
    drafts=(
        Draft(start='plain', model='gpt-6.1-sol', effort='high'),
        Draft(start='proportions', model='gpt-6.1-sol', effort='high'),
        Draft(start='plain', model='gpt-6.1-sol', effort='high'),
        Draft(start='proportions', model='gpt-6.1-sol', effort='high'),
        Draft(start='plain', model='gpt-6.1-sol', effort='high'),
        Draft(start='proportions', model='gpt-6.1-sol', effort='high'),
    ),
    steps=(
        Step(rewrites=(('gpt-6.1-sol', 'high'),), focus=('materials', 'shape')),
        Step(
            rewrites=(('gpt-6-astra', 'high'),),
            focus=('materials', 'shape'),
            fallback=('gpt-6.1-sol', 'xhigh'),
        ),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high')),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high')),
        Step(rewrites=(('gpt-6-astra', 'high'),), fallback=('gpt-6.1-sol', 'xhigh'), info=('gate',)),
        Step(rewrites=(('gpt-6.1-sol', 'max'),), fallback=('gpt-6.1-sol', 'xhigh'), info=('gate',)),
        Step(rewrites=(('gpt-6-astra', 'high'),), fallback=('gpt-6.1-sol', 'xhigh'), info=('gate',)),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high'), patch=True),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high'), patch=True),
        Step(rewrites=(('gpt-6.1-sol', 'xhigh'),), fallback=('gpt-6.1-sol', 'high'), patch=True),
        Step(rewrites=(('gpt-6.1-sol', 'high'),), patch=True),
        Step(rewrites=(('gpt-6.1-sol', 'high'),), patch=True),
        Step(rewrites=(('gpt-6.1-sol', 'high'),), patch=True),
    ),
    draft_keep=1,
    fill=0.999,
    render_cap=76,
)
