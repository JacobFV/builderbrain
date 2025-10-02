# BuilderBrain Jekyll Site

This directory contains a Jekyll-based technical blog site for BuilderBrain, featuring comprehensive documentation and technical deep-dives for mid/newbie developers.

## 🚀 Quick Start

### Local Development

1. **Install Ruby and Jekyll:**
   ```bash
   # Install Ruby (if not already installed)
   curl -fsSL https://github.com/rbenv/rbenv-installer/raw/HEAD/bin/rbenv-installer | bash

   # Install Jekyll
   gem install bundler jekyll
   ```

2. **Install dependencies:**
   ```bash
   bundle install
   ```

3. **Run locally:**
   ```bash
   bundle exec jekyll serve
   ```

4. **View the site:**
   Open `http://localhost:4000` in your browser

### GitHub Pages Deployment

The site is automatically deployed to GitHub Pages via GitHub Actions when changes are pushed to the `main` branch.

## 📁 Site Structure

```
/builderbrain/
├── _config.yml              # Jekyll configuration
├── _sass/
│   └── custom.scss         # Custom styling
├── index.html              # Homepage
├── posts/
│   └── _posts/             # Blog posts
│       ├── 2024-10-01-builderbrain-introduction.md
│       ├── 2024-10-02-dual-rail-architecture.md
│       ├── 2024-10-03-grammar-constraints.md
│       ├── 2024-10-04-plan-execution.md
│       ├── 2024-10-05-safety-invariants.md
│       └── 2024-10-06-training-methodology.md
├── Gemfile                 # Ruby dependencies
├── .github/
│   └── workflows/
│       └── jekyll.yml      # GitHub Actions workflow
└── README_JEKYLL.md        # This file
```

## 📝 Blog Post Series

The site features a comprehensive 6-part technical series explaining BuilderBrain:

1. **[Introduction](posts/_posts/2024-10-01-builderbrain-introduction.md)** - Overview and core problem
2. **[Dual-Rail Architecture](posts/_posts/2024-10-02-dual-rail-architecture.md)** - The heart of the system
3. **[Grammar Constraints](posts/_posts/2024-10-03-grammar-constraints.md)** - Structured output guarantees
4. **[Plan Execution](posts/_posts/2024-10-04-plan-execution.md)** - From plans to actions
5. **[Safety Invariants](posts/_posts/2024-10-05-safety-invariants.md)** - Risk energy and promotion gates
6. **[Training Methodology](posts/_posts/2024-10-06-training-methodology.md)** - Multi-objective optimization

Each post includes:
- ✅ Code examples and implementation details
- ✅ Mathematical foundations where relevant
- ✅ Real-world applications and use cases
- ✅ Challenges and solutions
- ✅ Next steps and further reading

## 🎨 Design Philosophy

The site is designed for **mid/newbie developers** who want to understand:

- **Technical depth** without overwhelming complexity
- **Practical examples** they can try themselves
- **Real-world applications** beyond just theory
- **Progressive disclosure** - concepts build on each other

## 🔧 Customization

### Adding New Posts

1. Create a new markdown file in `posts/_posts/` with the format:
   ```
   YYYY-MM-DD-post-title.md
   ```

2. Add front matter:
   ```yaml
   ---
   layout: post
   title: "Your Post Title"
   date: YYYY-MM-DD
   categories: category1 category2
   excerpt: "Brief description of the post"
   ---
   ```

3. Write your content using Markdown

### Styling

Custom styles are in `_sass/custom.scss`. The site uses:
- **Minima theme** as the base
- **Custom color scheme** (blue primary, clean whites)
- **Responsive design** for mobile and desktop
- **Syntax highlighting** for code blocks

### GitHub Actions

The workflow in `.github/workflows/jekyll.yml`:
- **Triggers** on pushes to `main` and pull requests
- **Builds** the Jekyll site using Ruby 3.1
- **Deploys** to GitHub Pages for production

## 🚀 Deployment

### Automatic Deployment
- Push changes to the `main` branch
- GitHub Actions will automatically build and deploy
- Site will be available at `https://your-username.github.io/builderbrain`

### Manual Deployment
```bash
# Build locally
bundle exec jekyll build

# Deploy to GitHub Pages
# (Requires GitHub Pages setup in repository settings)
```

## 📊 Analytics and Monitoring

### Site Analytics
- GitHub Pages provides basic traffic analytics
- Can be enhanced with Google Analytics or similar

### Performance
- Jekyll sites are fast and lightweight
- Static generation means no server-side processing
- CDN delivery through GitHub Pages

## 🔒 Security

- **Static site**: No server-side vulnerabilities
- **GitHub Pages**: Secure hosting with HTTPS
- **No external dependencies**: All content served from repository

## 🤝 Contributing

### Adding Content
1. Follow the existing post format
2. Use clear, technical but accessible language
3. Include code examples where helpful
4. Test links and formatting

### Site Improvements
- CSS improvements in `_sass/custom.scss`
- Layout changes in `index.html` and post templates
- New features or sections

## 📚 Resources

- [Jekyll Documentation](https://jekyllrb.com/docs/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [Minima Theme](https://github.com/jekyll/minima)
- [Markdown Guide](https://www.markdownguide.org/)

---

**The BuilderBrain Jekyll site makes complex AI concepts accessible through clear explanations, practical examples, and progressive technical depth. Perfect for developers who want to understand the cutting edge without getting lost in the details.**
